"""Convolution layers.

A model built from :class:`SeparableConvBlock` gets the separable-conv passes
(:mod:`flashim.quant.passes.separable_conv_block`): BatchNorm folding, the fused BF16 depthwise+pointwise kernel,
and MXFP8 / INT8 pointwise quantization with the depthwise conv as the quantizing producer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SeparableConvBlock(nn.Module):
    """``conv_pointwise(relu(norm(conv_depthwise(x))))``: depthwise ``k x k`` (same padding) -> BatchNorm (or identity
    once folded) -> ReLU -> pointwise 1x1, no residual. ``autocast_dtype`` runs the block under autocast (RoMaV2's
    refiners run in BF16)."""

    def __init__(
        self, channels: int, kernel_size: int = 5, *, autocast_dtype: torch.dtype | None = None, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.conv_depthwise = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm: nn.Module = nn.BatchNorm2d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv_pointwise = nn.Conv2d(channels, channels, 1)
        self.autocast_dtype = autocast_dtype

    def _forward(self, x: Tensor) -> Tensor:
        return self.conv_pointwise(self.relu(self.norm(self.conv_depthwise(x))))

    def forward(self, x: Tensor) -> Tensor:
        if self.autocast_dtype is None:
            return self._forward(x)
        with torch.autocast(x.device.type, self.autocast_dtype):
            return self._forward(x)
