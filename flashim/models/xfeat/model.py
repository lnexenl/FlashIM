"""XFeat (Potje et al., CVPR 2024): a 0.7M-parameter conv net at 1/8 resolution with a 65-way keypoint head, a
reliability heatmap and 64-d descriptors."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim.layers.conv import sequential_forward
from flashim.models import sparse
from flashim.models.xfeat import weights
from flashim.platforms import default_device
from flashim.utils import device_constant


class BasicLayer(nn.Module):
    """Conv2d -> BatchNorm (no affine) -> ReLU."""

    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(cout, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return sequential_forward(self.layer, x)  # the ReLU in a quantized conv's epilogue once BatchNorm is folded


def _bn_pairs(model: nn.Module) -> list[tuple[str, str]]:
    return [(f"{n}.layer.0", f"{n}.layer.1") for n, m in model.named_modules() if isinstance(m, BasicLayer)]


def _sample(x: Tensor, pos: Tensor, h: int, w: int, mode: str) -> Tensor:
    """Upstream ``InterpolateSparse2d``: sample ``x`` ``[1, C, ., .]`` at ``pos`` ``[N, 2]`` given in the ``h x w`` full
    resolution's pixel indices (normalized by ``size - 1``, ``align_corners=False``) -> ``[N, C]``."""
    grid = (2.0 * pos.float() / device_constant([w - 1, h - 1], pos, torch.float32) - 1.0).view(1, -1, 1, 2)
    return F.grid_sample(x.float(), grid, mode=mode, align_corners=False)[0, :, :, 0].t()


class XFeat(sparse.SparseExtractor):
    """Module paths follow ``xfeat.pt`` (``net.*`` upstream; loaded without the prefix); loads strictly. The
    ``fine_matcher`` MLP (upstream's semi-dense refinement) is kept for checkpoint parity and unused by the sparse path.

    ``detect`` (upstream ``detectAndCompute``): gray -> instance norm -> net -> keypoint logits softmax (64 cell
    positions) -> pixel heatmap -> 5x5 max-pool NMS above ``detection_threshold`` -> score = nearest(heatmap) x
    bilinear(reliability) -> top-k -> bicubic descriptors of the L2-normalized feature map. The two output 1x1 convs
    (keypoint logits, reliability) stay FP32 under reduced precision."""

    div_by = 32
    match_threshold = 0.82  # upstream ``match`` min_cossim
    fp32_modules = ("keypoint_head.3", "heatmap_head.2")

    def __init__(self, detection_threshold: float = 0.05) -> None:
        super().__init__()
        self.detection_threshold = detection_threshold
        self.norm = nn.InstanceNorm2d(1)
        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4), nn.Conv2d(1, 24, 1, stride=1, padding=0))
        self.block1 = nn.Sequential(BasicLayer(1, 4, stride=1), BasicLayer(4, 8, stride=2), BasicLayer(8, 8, stride=1), BasicLayer(8, 24, stride=2))
        self.block2 = nn.Sequential(BasicLayer(24, 24, stride=1), BasicLayer(24, 24, stride=1))
        self.block3 = nn.Sequential(BasicLayer(24, 64, stride=2), BasicLayer(64, 64, stride=1), BasicLayer(64, 64, 1, padding=0))
        self.block4 = nn.Sequential(BasicLayer(64, 64, stride=2), BasicLayer(64, 64, stride=1), BasicLayer(64, 64, stride=1))
        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2), BasicLayer(128, 128, stride=1), BasicLayer(128, 128, stride=1), BasicLayer(128, 64, 1, padding=0)
        )
        self.block_fusion = nn.Sequential(BasicLayer(64, 64, stride=1), BasicLayer(64, 64, stride=1), nn.Conv2d(64, 64, 1, padding=0))
        self.heatmap_head = nn.Sequential(BasicLayer(64, 64, 1, padding=0), BasicLayer(64, 64, 1, padding=0), nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0), BasicLayer(64, 64, 1, padding=0), BasicLayer(64, 64, 1, padding=0), nn.Conv2d(64, 65, 1)
        )
        layers = []
        for cin, cout in ((128, 512), (512, 512), (512, 512), (512, 512)):
            layers += [nn.Linear(cin, cout), nn.BatchNorm1d(cout, affine=False), nn.ReLU(inplace=True)]
        self.fine_matcher = nn.Sequential(*layers, nn.Linear(512, 64))

    @classmethod
    def from_pretrained(cls, *, checkpoint: str | Path | None = None, device: torch.device | str | None = None,
                        **kwargs) -> XFeat:
        with torch.device("meta"):
            model = cls(**kwargs)
        model.load_state_dict(weights.load_state_dict(checkpoint), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return _bn_pairs(self)

    @staticmethod
    def _unfold2d(x: Tensor, ws: int = 8) -> Tensor:
        b, c, h, w = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(b, c, h // ws, w // ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(b, -1, h // ws, w // ws)

    def _head(self, head: nn.Sequential, x: Tensor) -> Tensor:
        """``head`` with its final 1x1 conv (+ sigmoid) run in FP32."""
        x = head[0:-1](x) if isinstance(head[-1], nn.Conv2d) else head[0:-2](x)
        last = head[-1:] if isinstance(head[-1], nn.Conv2d) else head[-2:]
        return last(x.float() if last[0].weight.dtype == torch.float32 else x)

    def dense(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(feats [B, 64, H/8, W/8], keypoint logits [B, 65, H/8, W/8], reliability [B, 1, H/8, W/8])``."""
        gray = self.norm(image.mean(dim=1, keepdim=True)).to(self.compute_dtype)
        x1 = self.block1(gray)
        x2 = self.block2(x1 + self.skip1(gray))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        size = x3.shape[-2:]
        x4 = F.interpolate(x4, size, mode="bilinear")
        x5 = F.interpolate(x5, size, mode="bilinear")
        feats = self.block_fusion(x3 + x4 + x5)
        heatmap = self._head(self.heatmap_head, feats)
        keypoints = self._head(self.keypoint_head, self._unfold2d(gray, 8))
        return feats.float(), keypoints.float(), heatmap.float()

    @staticmethod
    def keypoint_heatmap(logits: Tensor) -> Tensor:
        """65-way logits ``[B, 65, h, w]`` -> pixel heatmap ``[B, 1, 8h, 8w]`` (softmax over the 64 cell positions)."""
        scores = F.softmax(logits, 1)[:, :64]
        b, _, h, w = scores.shape
        return scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, 1, h * 8, w * 8)

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        h, w = image.shape[-2:]
        feats, logits, reliability = self.dense(image)
        feats = F.normalize(feats, dim=1)
        heatmap = self.keypoint_heatmap(logits)
        pos = (heatmap == F.max_pool2d(heatmap, 5, 1, 2)) & (heatmap > self.detection_threshold)
        ys, xs = pos[0, 0].nonzero(as_tuple=True)
        keypoints = torch.stack((xs, ys), -1)
        scores = _sample(heatmap, keypoints, h, w, "nearest")[:, 0] * _sample(reliability, keypoints, h, w, "bilinear")[:, 0]
        keypoints, scores, _ = sparse.select_top_k(keypoints, scores, top_k)
        valid = scores > 0
        keypoints, scores = keypoints[valid], scores[valid]
        descriptors = F.normalize(_sample(feats, keypoints, h, w, "bicubic"), dim=-1)
        return keypoints.float(), scores, descriptors
