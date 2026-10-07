"""ALIKE (Zhao et al., 2022): ALIKED's predecessor — plain-conv encoder, multi-scale fusion, one 1x1 head giving a
descriptor map and a keypoint score map at the input resolution."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import resnet

from flashim import ops
from flashim.layers.conv import conv_relu
from flashim.models import sparse
from flashim.models.alike import weights
from flashim.models.alike.config import VARIANTS, ALIKEConfig
from flashim.platforms import default_device


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, gate: nn.Module) -> None:
        super().__init__()
        self.gate = gate
        self.conv1, self.bn1 = resnet.conv3x3(cin, cout), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = resnet.conv3x3(cout, cout), nn.BatchNorm2d(cout)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.gate, nn.ReLU):  # ReLU in the (quantized) convolutions' epilogue when they support it
            return conv_relu(self.conv2, conv_relu(self.conv1, x, self.bn1, feeds=self.conv2), self.bn2)
        return self.gate(self.bn2(self.conv2(self.gate(self.bn1(self.conv1(x))))))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, gate: nn.Module) -> None:
        super().__init__()
        self.gate = gate
        self.conv1, self.bn1 = resnet.conv3x3(cin, cout), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = resnet.conv3x3(cout, cout), nn.BatchNorm2d(cout)
        self.downsample = nn.Conv2d(cin, cout, 1)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.gate, nn.ReLU):
            hidden = conv_relu(self.conv1, x, self.bn1, feeds=self.conv2)
        else:
            hidden = self.gate(self.bn1(self.conv1(x)))
        return self.gate(self.bn2(self.conv2(hidden)) + self.downsample(x))


class ALIKE(sparse.SparseExtractor):
    """Module paths follow the released checkpoints (loads strictly). ``detect``: dense score / descriptor maps ->
    ``simple_nms`` (radius 2) -> border removal -> top-k (or ``scores > scores_th``, at most ``n_limit``) ->
    integer-pixel keypoints; descriptors are read at those pixels and L2-normalized. The last 1x1 head
    (``convhead2``: descriptor + score logit) stays FP32 under reduced precision so the score map keeps the
    resolution NMS ties depend on."""

    div_by = 32
    fp32_modules = ("convhead2",)

    def __init__(self, config: ALIKEConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or VARIANTS["alike-n"]
        self.match_threshold = cfg.match_threshold
        self.gate = nn.ReLU(inplace=True)
        self.block1 = ConvBlock(3, cfg.c1, self.gate)
        self.block2 = ResBlock(cfg.c1, cfg.c2, self.gate)
        self.block3 = ResBlock(cfg.c2, cfg.c3, self.gate)
        self.block4 = ResBlock(cfg.c3, cfg.c4, self.gate)
        quarter = cfg.dim // 4
        self.conv1, self.conv2, self.conv3 = (resnet.conv1x1(c, quarter) for c in (cfg.c1, cfg.c2, cfg.c3))
        self.conv4 = resnet.conv1x1(cfg.dim, quarter)
        if not cfg.single_head:
            self.convhead1 = resnet.conv1x1(cfg.dim, cfg.dim)
        self.convhead2 = resnet.conv1x1(cfg.dim, cfg.dim + 1)

    @classmethod
    def from_pretrained(cls, config: ALIKEConfig | None = None, *, variant: str = "alike-n",
                        checkpoint: str | Path | None = None, device: torch.device | str | None = None) -> ALIKE:
        with torch.device("meta"):
            model = cls(config or VARIANTS[variant])
        model.load_state_dict(weights.load_state_dict(variant, checkpoint), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return [(f"block{i}.conv{j}", f"block{i}.bn{j}") for i in range(1, 5) for j in (1, 2)]

    def dense(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """``(scores [B, 1, H, W] in (0, 1), descriptors [B, dim, H, W] unnormalized)`` at the input resolution."""
        fused = self.head_input(image)
        out = self.convhead2(fused.to(self.convhead2.weight.dtype)).float()
        return torch.sigmoid(out[:, -1:]), out[:, :-1]

    def head_input(self, image: Tensor) -> Tensor:
        """The multi-scale features ``convhead2`` reads (compute dtype)."""
        x1 = self.block1(image.to(self.compute_dtype))
        x2 = self.block2(F.max_pool2d(x1, 2))
        x3 = self.block3(F.max_pool2d(x2, 4))
        x4 = self.block4(F.max_pool2d(x3, 4))
        size = image.shape[-2:]
        levels = [self.gate(conv(x)) for conv, x in zip((self.conv1, self.conv2, self.conv3, self.conv4), (x1, x2, x3, x4))]
        fused = torch.cat(
            [levels[0]] + [F.interpolate(x, size=size, mode="bilinear", align_corners=True) for x in levels[1:]], dim=1
        )
        if not self.config.single_head:
            fused = self.gate(self.convhead1(fused))
        return fused

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        # convhead2 split by output row: the score row dense (FP32, read straight from the compute-dtype features), the
        # descriptor rows only at the selected keypoints (the dense FP32 descriptor map is only ever indexed there)
        fused = self.head_input(image)
        weight, bias = self.convhead2.weight, self.convhead2.bias
        scores_map = torch.sigmoid(ops.conv2d_single_output(fused, weight[-1:], None if bias is None else bias[-1:]))
        _, _, h, w = scores_map.shape
        nms = sparse.simple_nms(scores_map, 2)
        r = cfg.radius
        nms[:, :, : r + 1], nms[:, :, :, : r + 1], nms[:, :, h - r :], nms[:, :, :, w - r :] = 0, 0, 0, 0
        flat = nms.reshape(-1)
        if top_k is not None and top_k > 0:
            index = torch.topk(flat, min(top_k, flat.numel())).indices
        else:
            mask = flat > cfg.scores_th
            if not mask.any():
                mask = flat > scores_map.mean()
            index = mask.nonzero()[:, 0]
            if len(index) > cfg.n_limit:
                index = index[flat[index].sort(descending=True)[1][: cfg.n_limit]]
        keypoints = torch.stack((index % w, index // w), dim=1)
        at_keypoints = fused[0, :, keypoints[:, 1], keypoints[:, 0]].t().float()
        descriptors = F.linear(at_keypoints, weight[:-1, :, 0, 0].float(), None if bias is None else bias[:-1].float())
        descriptors = F.normalize(descriptors, dim=-1)
        return keypoints.float(), scores_map.reshape(-1)[index], descriptors
