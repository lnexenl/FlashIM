"""RoMa (v1) encoder: VGG19-BN fine feature pyramid (strides 1, 2, 4, 8) + frozen DINOv2 ViT-L/14 (stride 14)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim.models.dedode.blocks import VGG, DinoV2


class CNNandDinov2(nn.Module):
    """Parameter names follow RoMa's checkpoint (``cnn.layers.*``) plus ``dinov2.*`` for the frozen DINOv2 ViT-L/14."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn = VGG("19")
        self.dinov2 = DinoV2()
        self.channels_last = False

    def batchnorm_pairs(self, prefix: str = "") -> list[tuple[str, str]]:
        return self.cnn.batchnorm_pairs(f"{prefix}cnn.")

    def _cnn_dtype(self) -> torch.dtype:
        first = self.cnn.layers[0]
        if hasattr(first, "weight"):
            return first.weight.dtype
        return torch.bfloat16

    def forward(self, x: Tensor, *, upsample: bool = False) -> dict[int, Tensor]:
        x_cnn = x.to(self._cnn_dtype())
        if self.channels_last:
            x_cnn = x_cnn.to(memory_format=torch.channels_last)
        feats, _ = self.cnn(x_cnn)
        pyramid: dict[int, Tensor] = {1: feats[0], 2: feats[1], 4: feats[2], 8: feats[3]}
        if not upsample:
            pyramid[16] = self.dinov2(x)
        return pyramid
