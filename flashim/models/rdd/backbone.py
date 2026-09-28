"""ConvNeXt backbone (module names follow the RDD checkpoint, minus the wrapper prefixes)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """LayerNorm over ``channels_last`` inputs or over dim 1 of ``channels_first`` inputs (float32 math)."""

    def __init__(self, dim: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, (self.dim,), self.weight, self.bias, self.eps)
        dtype = x.dtype
        x = x.float()
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (self.weight.float()[:, None, None] * x + self.bias.float()[:, None, None]).to(dtype)


class Block(nn.Module):
    """dwconv7x7 -> LayerNorm -> Linear(4x) -> GELU -> Linear -> layer scale, residual."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        y = (self.gamma * y).permute(0, 3, 1, 2)
        return x + y


class ConvNeXt(nn.Module):
    """Feature pyramid: returns the output of each of the four stages."""

    def __init__(self, depths: tuple[int, ...], dims: tuple[int, ...]) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0], data_format="channels_first"))]
        )
        for i in range(len(dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )
        self.stages = nn.ModuleList(nn.Sequential(*[Block(dim) for _ in range(depth)]) for depth, dim in zip(depths, dims))

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            features.append(x)
        return features
