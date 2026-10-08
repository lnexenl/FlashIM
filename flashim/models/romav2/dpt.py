# Follows code that RoMaV2 took from Meta (DINOv3 License Agreement, Copyright (c) Meta Platforms, Inc. and affiliates);
# see THIRD_PARTY_NOTICES.md.
"""DPT dense-prediction head as used by the RoMaV2 matcher (inference only).

Everything up to ``output_conv1`` and the final upsample runs under BF16 autocast;
``output_conv2`` runs in FP32. Note the shared in-place ReLU inside the residual
units: it also rectifies the unit's input before the skip add (upstream behavior).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.activation(self.conv1(self.activation(x))))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True) -> None:
        super().__init__()
        activation = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(features, features, 1)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(self, *xs: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        output = xs[0]
        if self.has_residual:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        if size is None:
            size = (output.shape[-2] * 2, output.shape[-1] * 2)
        output = F.interpolate(output, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(output)


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        out_dim: int,
        patch_size: int = 16,
        features: int = 256,
        out_channels: tuple[int, ...] = (256, 512, 1024, 1024),
        down_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.down_ratio = down_ratio
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(nn.Conv2d(dim_in, oc, 1) for oc in out_channels)
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        scratch = nn.Module()
        for i, oc in enumerate(out_channels, start=1):
            setattr(scratch, f"layer{i}_rn", nn.Conv2d(oc, features, 3, padding=1, bias=False))
        scratch.refinenet1 = FeatureFusionBlock(features)
        scratch.refinenet2 = FeatureFusionBlock(features)
        scratch.refinenet3 = FeatureFusionBlock(features)
        scratch.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, out_dim, 1)
        )
        self.scratch = scratch

    def forward(self, tokens: list[Tensor]) -> Tensor:
        """``tokens``: list of ``[B, h, w, D]``; returns ``[B, 4h, 4w, out_dim]`` (FP32)."""
        batch, h, w, _ = tokens[0].shape
        factor = len(self.out_channels) // len(tokens)
        tokens = [t for t in tokens for _ in range(factor)]
        with torch.autocast(tokens[0].device.type, torch.bfloat16):
            out = []
            for index, x in enumerate(tokens):
                x = self.norm(x.reshape(batch, -1, x.shape[-1]))
                x = x.permute(0, 2, 1).reshape(batch, x.shape[-1], h, w)
                out.append(self.resize_layers[index](self.projects[index](x)))
            fused = self._fuse(out)
            size = (h * self.patch_size // self.down_ratio, w * self.patch_size // self.down_ratio)
            fused = F.interpolate(fused, size=size, mode="bilinear", align_corners=True)
        return self.scratch.output_conv2(fused.float()).permute(0, 2, 3, 1)

    def _fuse(self, features: list[Tensor]) -> Tensor:
        s = self.scratch
        l1, l2, l3, l4 = (getattr(s, f"layer{i}_rn")(f) for i, f in enumerate(features, start=1))
        out = s.refinenet4(l4, size=l3.shape[2:])
        out = s.refinenet3(out, l3, size=l2.shape[2:])
        out = s.refinenet2(out, l2, size=l1.shape[2:])
        out = s.refinenet1(out, l1)
        return s.output_conv1(out)
