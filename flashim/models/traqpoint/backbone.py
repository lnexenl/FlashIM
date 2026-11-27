"""DINOv3 ConvNeXt-Base backbone for TraqPoint (returns all four stages, with final LayerNorm on stage 3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm over dim 1 of a ``[B, C, H, W]`` tensor (float32 statistics in NCHW)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if x.is_contiguous(memory_format=torch.channels_last) and x.shape[-2:] != (1, 1):
            return F.layer_norm(x.permute(0, 2, 3, 1), (self.dim,), self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        dtype = x.dtype
        x = x.float()
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (self.weight.float()[:, None, None] * x + self.bias.float()[:, None, None]).to(dtype)


class ConvNeXtBlock(nn.Module):
    """dwconv7x7 -> LayerNorm -> Linear(4x) -> GELU -> Linear -> layer scale, residual."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def fold_gamma(self) -> bool:
        """Fold LayerScale ``gamma`` into ``pwconv2`` before BF16 cast / quantization."""
        if self.gamma is None:
            return False
        with torch.no_grad():
            g = self.gamma.to(self.pwconv2.weight.dtype)
            self.pwconv2.weight.mul_(g[:, None])
            if self.pwconv2.bias is not None:
                self.pwconv2.bias.mul_(g.to(self.pwconv2.bias.dtype))
        self.register_parameter("gamma", None)
        return True

    def forward(self, x: Tensor) -> Tensor:
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        if self.gamma is not None:
            y = self.gamma * y
        return x + y.permute(0, 3, 1, 2)


class DINOv3ConvNeXt(nn.Module):
    """DINOv3 ConvNeXt pyramid: returns ``[s0, s1, s2, norm(s3)]`` matching ``get_intermediate_layers(n=[0,1,2,3])``."""

    channels_last: bool = False

    def __init__(self, depths: tuple[int, ...] = (3, 3, 27, 3), dims: tuple[int, ...] = (128, 256, 512, 1024)) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0], eps=1e-6))]
            + [
                nn.Sequential(LayerNorm2d(dims[i], eps=1e-6), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
                for i in range(len(dims) - 1)
            ]
        )
        self.stages = nn.ModuleList(
            nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)]) for depth, dim in zip(depths, dims)
        )
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        features = []
        last = len(self.stages) - 1
        for idx, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = stage(downsample(x))
            if idx == last:
                x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            features.append(x)
        return features
