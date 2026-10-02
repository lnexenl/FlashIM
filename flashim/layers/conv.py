"""Convolution layers.

A model built from :class:`SeparableConvBlock` gets the separable-conv passes
(:mod:`flashim.quant.passes.separable_conv_block`): BatchNorm folding, the fused BF16 depthwise+pointwise kernel,
and MXFP8 / INT8 pointwise quantization with the depthwise conv as the quantizing producer.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn


class SeparableConvBlock(nn.Module):
    """``conv_pointwise(relu(norm(conv_depthwise(x))))``: depthwise ``k x k`` (same padding) -> BatchNorm (or identity
    once folded) -> ReLU -> pointwise 1x1, no residual. ``autocast_dtype`` runs the block under autocast (RoMaV2's
    refiners run in BF16)."""

    pointwise_name = "conv_pointwise"  # child name of the pointwise conv: what a quantization scheme targets
    fuse_quantized = True  # quantizing ``pointwise_name`` replaces the whole block by its fused INT8 / MXFP8 module

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


class IndexedSeparableConvBlock(SeparableConvBlock):
    """:class:`SeparableConvBlock` whose children carry ``nn.Sequential`` names (``0`` depthwise, ``1`` norm, ``2``
    ReLU, ``3`` pointwise), for checkpoints of models that wrote the block as
    ``nn.Sequential(conv, bn, relu, conv)`` (DeDoDe's / RIPE's ``ConvRefiner``): the state dict keeps its upstream
    keys, the passes still see the layer type. ``fuse_quantized`` is off: a quantization scheme targeting ``3`` packs
    that 1x1 conv like any other (the fused INT8 / MXFP8 block modules only cover 32 / 128 / 512 channels)."""

    pointwise_name = "3"
    fuse_quantized = False
    _INDEX: ClassVar[dict[str, str]] = {"conv_depthwise": "0", "norm": "1", "relu": "2", "conv_pointwise": "3"}

    def __getattr__(self, name: str):
        index = type(self)._INDEX.get(name)
        if index is not None:
            return self._modules[index]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(type(self)._INDEX.get(name, name), value)
