"""FlexGlue's DINOv3-ConvNeXt dense descriptor (backbone + linear-conv head).

A small ConvNeXt that turns an image into a stride-8 ``input_dim``-wide feature map (the matcher's search map) and
L2-normalized ``input_dim`` descriptors at keypoints (the layer-0 readout, fused by upstream into ``descriptor_head``).
Ported from cvg/FlexGlue's ``descriptor.py``; names match the ``extractor.*`` checkpoint entries.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _to_hwc(x: Tensor) -> Tensor:
    """``(b, c, h, w)`` -> ``(b, h, w, c)``."""
    return x.permute(0, 2, 3, 1)


def _to_chw(x: Tensor) -> Tensor:
    """``(b, h, w, c)`` -> ``(b, c, h, w)``."""
    return x.permute(0, 3, 1, 2)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of a ``[B, C, H, W]`` tensor."""

    def forward(self, x: Tensor) -> Tensor:
        return _to_chw(super().forward(_to_hwc(x)))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(y)))
        return x + y.permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    """DINOv3 ConvNeXt backbone, returns the unnormalized features of all 4 stages."""

    def __init__(self, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm2d(dims[0], 1e-6))
            ]
            + [
                nn.Sequential(LayerNorm2d(dims[i], 1e-6), nn.Conv2d(dims[i], dims[i + 1], 2, 2))
                for i in range(3)
            ]
        )
        self.stages = nn.ModuleList(
            nn.Sequential(*[ConvNeXtBlock(d) for _ in range(n)]) for n, d in zip(depths, dims)
        )

    def forward(self, x: Tensor) -> list:
        features = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            features.append(x)
        return features


class LinearConvHead(nn.Module):
    """Stride-8 features: unshuffled conv stem on stride 4 plus linear strides 8-32."""

    def __init__(self, in_dims=(128, 256, 512, 1024), dim=512, out_dim=256) -> None:
        super().__init__()
        fine_in, *coarse_in = in_dims
        c4 = dim // 4
        self.stem = nn.Sequential(
            LayerNorm2d(fine_in),
            nn.Conv2d(fine_in, c4, 3, padding=1),
            LayerNorm2d(c4),
            nn.GELU(),
        )
        self.norms = nn.ModuleList(LayerNorm2d(c) for c in coarse_in)
        self.fine_proj = nn.Conv2d(4 * c4, dim, 1)
        self.projs = nn.ModuleList(nn.Conv2d(c, dim, 1, bias=False) for c in coarse_in)
        self.mlp = nn.Sequential(
            LayerNorm2d(dim), nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.down = nn.Conv2d(dim, out_dim, 1)
        self.out_norm = LayerNorm2d(out_dim)

    def forward(self, features: list) -> Tensor:
        fine, *coarse = features
        h, w = coarse[0].shape[-2:]
        x = self.fine_proj(F.pixel_unshuffle(self.stem(fine), 2))
        for norm, proj, c in zip(self.norms, self.projs, coarse):
            c = proj(norm(c))  # 1x1 commutes with bilinear, so project first
            if c.shape[-2:] != (h, w):
                c = F.interpolate(c, (h, w), mode="bilinear", align_corners=False)
            x = x + c
        return self.out_norm(self.down(x + self.mlp(x)))


class DINOv3Descriptor(nn.Module):
    """Describes images densely (search map) and keypoints sparsely (descriptors)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ConvNeXt()
        self.head = LinearConvHead()
        # fused layer-0 matcher head + input_proj + to_search readout: 256 -> 64, L2-normalized by the caller
        self.descriptor_head = nn.Linear(256, 64)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None], persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[:, None, None], persistent=False)

    def forward(self, image: Tensor) -> Tensor:
        """Dense ``input_dim``-wide stride-8 features ``[B, C, h/8, w/8]`` of ``[B, 3, H, W]`` in ``[0, 1]``."""
        return self.head(self.backbone((image - self.mean) / self.std))
