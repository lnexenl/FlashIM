"""DISK (Tyszkiewicz et al., 2020): a 5-level U-Net of 5x5 convolutions (InstanceNorm -> PReLU -> conv) producing a
128-d descriptor map and a keypoint heatmap at the input resolution. Follows kornia's ``DISK`` module layout."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models import sparse
from flashim.models.disk import weights
from flashim.platforms import default_device
from flashim.utils import set_submodule


class Conv(nn.Sequential):
    """``conv(prelu(instance_norm(x)))`` (norm / gate skipped for the input conv); the input is cast to the conv's
    weight dtype, so a conv left in FP32 under a BF16 compute dtype (the output head) receives FP32."""

    def __init__(self, cin: int, cout: int, size: int, skip_norm_and_gate: bool = False) -> None:
        norm = nn.Sequential() if skip_norm_and_gate else nn.InstanceNorm2d(cin)
        gate = nn.Sequential() if skip_norm_and_gate else nn.PReLU(cin)
        super().__init__(norm, gate, nn.Sequential(), nn.Conv2d(cin, cout, size, padding="same", bias=True))

    def prologue(self, x: Tensor) -> Tensor:
        """``gate(norm(x))``: the convolution's input."""
        return self[2](self[1](self[0](x)))

    def forward(self, x: Tensor) -> Tensor:
        x = self.prologue(x)
        conv = self[3]
        return conv(x.to(conv.weight.dtype) if hasattr(conv, "weight") else x)


class DownBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, size: int, is_first: bool) -> None:
        super().__init__(nn.Sequential() if is_first else nn.AvgPool2d(2), Conv(cin, cout, size, skip_norm_and_gate=is_first))


class UpBlock(nn.Module):
    def __init__(self, bottom: int, horizontal: int, cout: int, size: int) -> None:
        super().__init__()
        self.conv = Conv(bottom + horizontal, cout, size)

    def conv_input(self, bottom: Tensor, horizontal: Tensor) -> Tensor:
        # A quantized conv leaves `bottom` channels-last; the skip input is NCHW. Mixed layouts send the concat to its
        # slow generic kernel and the upsample to the NHWC one, so the small (low-resolution) input goes NCHW first.
        big = F.interpolate(bottom.contiguous(), scale_factor=2, mode="bilinear", align_corners=False)
        return torch.cat([big, horizontal.to(big.dtype)], dim=1)

    def forward(self, bottom: Tensor, horizontal: Tensor) -> Tensor:
        return self.conv(self.conv_input(bottom, horizontal))


class SplitHeadConv(nn.Module):
    """``conv`` split by output channel into ``desc`` (the first ``desc_dim`` channels, follows the model's compute
    dtype) and ``heat`` (the last channel, always FP32 -- see :attr:`DISK.fp32_modules`). Only the 1-channel heatmap
    needs FP32 for strict-NMS resolution; keeping the whole ``desc_dim + 1``-channel conv FP32 (as one ``nn.Conv2d``)
    paid for that on every channel. Measured: this conv is the single largest kernel in DISK's forward, ~70% of
    total extraction time at 1600px (``benchmarks/profile/extractor_profile.py``), because it runs at the input
    image's full resolution with the U-Net's highest channel count; splitting it moves the descriptor channels onto
    the same fast BF16/tensor-core path as every other conv, leaving only the free (1-channel) heatmap GEMM at
    FP32. Output channel order matches the original conv (descriptors first, heatmap last), so :meth:`DISK.dense`
    needs no change."""

    def __init__(self, conv: nn.Conv2d, desc_dim: int) -> None:
        super().__init__()
        heat_dim = conv.out_channels - desc_dim
        device = conv.weight.device
        self.desc = nn.Conv2d(conv.in_channels, desc_dim, conv.kernel_size, padding=conv.padding, device=device)
        self.heat = nn.Conv2d(conv.in_channels, heat_dim, conv.kernel_size, padding=conv.padding, device=device)
        with torch.no_grad():
            self.desc.weight.copy_(conv.weight[:desc_dim])
            self.heat.weight.copy_(conv.weight[desc_dim:])
            if conv.bias is None:
                self.desc.bias, self.heat.bias = None, None
            else:
                self.desc.bias.copy_(conv.bias[:desc_dim])
                self.heat.bias.copy_(conv.bias[desc_dim:])

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.desc(x.to(self.desc.weight.dtype)), self.heat(x.float())], dim=1)


class Unet(nn.Module):
    def __init__(self, in_features: int, down: list[int], up: list[int], size: int = 5) -> None:
        super().__init__()
        dims = [in_features, *down]
        self.path_down = nn.ModuleList(DownBlock(a, b, size, i == 0) for i, (a, b) in enumerate(pairwise(dims)))
        bottom = [down[-1], *up]
        horizontal = dims[-2::-1]
        self.path_up = nn.ModuleList(UpBlock(a, b, c, size) for a, b, c in zip(bottom, horizontal, up))
        self.divisor = 2 ** len(up)

    def forward(self, image: Tensor, head_input: bool = False) -> Tensor:
        """The output map; with ``head_input`` the input of the last convolution instead (normalized and gated)."""
        features = [image]
        for layer in self.path_down:
            features.append(layer(features[-1]))
        x = features[-1]
        last = len(self.path_up) - 1
        for i, (layer, horizontal) in enumerate(zip(self.path_up, features[-2::-1])):
            if head_input and i == last:
                return layer.conv.prologue(layer.conv_input(x, horizontal))
            x = layer(x, horizontal)
        return x


class DISK(sparse.SparseExtractor):
    """``detect``: U-Net -> heatmap (the raw logit is the score) -> strict 5x5 NMS (a pixel must exceed all 24
    neighbours; a 2-pixel border is never a keypoint) and ``heatmap > 0`` -> top-k -> integer-pixel keypoints,
    L2-normalized descriptors read at them. The last conv (descriptor + heatmap logits) is split by
    :meth:`split_head_conv` (a ``pre_quant`` pass, see ``spec.py``) into two smaller convs instead of one
    ``desc_dim + 1``-channel one; both stay FP32 (matching the pre-split precision exactly), which alone is
    measurably faster (better GEMM shapes for cuDNN to pick a kernel for) without changing what's computed. Only
    the 1-channel ``heat`` half actually needs FP32 for the strict NMS comparisons to keep FP32 resolution; letting
    ``desc`` follow the model's compute dtype like every other conv is a further, real speedup but also a real
    precision change to descriptors that were previously incidentally FP32-protected -- see the accuracy note in
    ``spec.py`` before enabling it."""

    div_by = 16
    fp32_modules = ("unet.path_up.3.conv.3.desc", "unet.path_up.3.conv.3.heat")

    def __init__(self, desc_dim: int = 128, nms_window: int = 5, score_threshold: float = 0.0) -> None:
        super().__init__()
        self.desc_dim, self.nms_window, self.score_threshold = desc_dim, nms_window, score_threshold
        self.unet = Unet(3, down=[16, 32, 64, 64, 64], up=[64, 64, 64, desc_dim + 1], size=5)

    @classmethod
    def from_pretrained(cls, *, checkpoint: str | Path | None = None, device: torch.device | str | None = None,
                        **kwargs) -> DISK:
        with torch.device("meta"):
            model = cls(**kwargs)
        model.load_state_dict(weights.load_state_dict(checkpoint), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

    def split_head_conv(self) -> str:
        """Replace the output head with :class:`SplitHeadConv` (exact: same weights, same math, split by output
        channel). Run before any compute-dtype cast, so both halves start from the checkpoint's FP32 weights."""
        path = "unet.path_up.3.conv.3"
        set_submodule(self, path, SplitHeadConv(self.get_submodule(path), self.desc_dim))
        return path

    def dense(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """``(heatmap [B, 1, H, W], descriptors [B, desc_dim, H, W])`` (FP32, descriptors unnormalized)."""
        out = self.unet(image.to(self.compute_dtype)).float()
        return out[:, self.desc_dim :], out[:, : self.desc_dim]

    def nms_mask(self, heatmap: Tensor) -> Tensor:
        """Pixels strictly greater than every other pixel of their ``window x window`` neighbourhood (border excluded)."""
        r = self.nms_window // 2
        h, w = heatmap.shape[-2:]
        centre = heatmap[..., r : h - r, r : w - r]
        keep = torch.ones_like(centre, dtype=torch.bool)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy or dx:
                    keep &= centre > heatmap[..., r + dy : h - r + dy, r + dx : w - r + dx]
        mask = torch.zeros_like(heatmap, dtype=torch.bool)
        mask[..., r : h - r, r : w - r] = keep
        return mask

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        head = self.unet.path_up[-1].conv[3]
        if isinstance(head, SplitHeadConv):  # the descriptor half of the head only at the selected keypoints
            features = self.unet(image.to(self.compute_dtype), head_input=True)
            # FP32 heatmap straight from the compute-dtype features (no FP32 copy of them): full FP32 accumulation in
            # the Triton impl, more exact than the cuDNN TF32 the dense head used (max rel 2e-6 vs 2e-4 of float64)
            heatmap = ops.conv2d_single_output(features, head.heat.weight, head.heat.bias)
        else:
            heatmap, descriptor_map = self.dense(image)
        selected = self.nms_mask(heatmap)[0, 0] & (heatmap[0, 0] > self.score_threshold)
        ys, xs = selected.nonzero(as_tuple=True)
        keypoints, scores, _ = sparse.select_top_k(torch.stack((xs, ys), -1), heatmap[0, 0, ys, xs], top_k)
        if isinstance(head, SplitHeadConv):
            # same FP32 weights as the dense head; measured 13.9 ms (dense, cuDNN TF32) -> 0.36 ms at 4096 keypoints
            # on sm_120, plus the 129-channel FP32 concat (3.3 ms) the dense output needed; full FP32 (not TF32) here
            descriptors = ops.conv2d_at_points(features, head.desc.weight, head.desc.bias, keypoints).float()
        else:
            descriptors = descriptor_map[0][:, keypoints[:, 1], keypoints[:, 0]].t()
        return keypoints.float(), scores, F.normalize(descriptors, dim=-1)
