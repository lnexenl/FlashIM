"""Convolution layers.

A model built from :class:`SeparableConvBlock` gets the separable-conv passes
(:mod:`flashim.quant.passes.separable_conv_block`): BatchNorm folding, the fused BF16 depthwise+pointwise kernel,
and MXFP8 / INT8 pointwise quantization with the depthwise conv as the quantizing producer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def chains(producer: nn.Module, consumer: nn.Module | None) -> bool:
    """Whether ``producer(x, relu=True, quantize_for=consumer)`` hands ``consumer`` its quantized input directly
    (quantized convolutions implement ``feeds``); the result must then go to ``consumer`` and nowhere else."""
    feeds = getattr(producer, "feeds", None)
    return consumer is not None and feeds is not None and feeds(consumer)


def conv_relu(conv: nn.Module, x: Tensor, norm: nn.Module | None = None, feeds: nn.Module | None = None) -> Tensor:
    """``relu(norm(conv(x)))``, with the ReLU in the convolution's epilogue when the module supports it (quantized
    convolutions set ``fuses_relu``) and ``norm`` is folded away (``None`` / :class:`nn.Identity`); otherwise the plain
    in-place ReLU. Same bits either way: ReLU commutes with the epilogue's BF16 rounding."""
    if getattr(conv, "fuses_relu", False) and (norm is None or isinstance(norm, nn.Identity)):
        return conv(x, relu=True, quantize_for=feeds) if chains(conv, feeds) else conv(x, relu=True)
    return F.relu(conv(x) if norm is None else norm(conv(x)), inplace=True)


def fused_relu_end(layers: Sequence[nn.Module], i: int) -> int | None:
    """Index of the :class:`nn.ReLU` that ``layers[i]`` can apply in its epilogue (``layers[i]`` fuses ReLU and only
    folded norms, :class:`nn.Identity`, sit in between), else ``None``."""
    if not getattr(layers[i], "fuses_relu", False):
        return None
    j = i + 1
    while j < len(layers) and isinstance(layers[j], nn.Identity):
        j += 1
    return j if j < len(layers) and isinstance(layers[j], nn.ReLU) else None


def sequential_forward(layers: Sequence[nn.Module], x: Tensor) -> Tensor:
    """``nn.Sequential`` semantics over ``layers``, running each ``conv -> Identity* -> ReLU`` as ``conv(x, relu=True)``
    (:func:`fused_relu_end`): measured as a separate elementwise pass per quantized conv (DeDoDe int8: 3.6 ms of ReLU
    kernels per extract, ``benchmarks/profile/timeline_audit.py``)."""
    i = 0
    while i < len(layers):
        end = fused_relu_end(layers, i)
        if end is None:
            x = layers[i](x)
            i += 1
        else:
            consumer = layers[end + 1] if end + 1 < len(layers) else None
            x = layers[i](x, relu=True, quantize_for=consumer) if chains(layers[i], consumer) else layers[i](x, relu=True)
            i = end + 1
    return x


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
