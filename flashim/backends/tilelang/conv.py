"""TileLang depthwise 5x5 impls: ``custom_op`` wrappers and registrations.

The kernel builders (``_conv_kernels``, which imports TileLang/TVM) load on the first call, so loading this backend
stays cheap. ``depthwise5x5_relu_mxfp8`` has the same math and summation order as the Triton kernel (bit-identical
output); input rows are staged in shared memory by a warp-specialized TMA pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


def _kernels():
    from flashim.backends.tilelang import _conv_kernels

    return _conv_kernels


@torch.library.custom_op("flashim::depthwise5x5_relu_mxfp8_tilelang", mutates_args=())
def _depthwise5x5_relu_mxfp8(value: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    if value.data_ptr() % 16:  # TMA needs a 16-byte aligned base (checked here: not traceable)
        value = value.clone(memory_format=torch.channels_last)
    batch, channels, height, width = value.shape
    rows = batch * height * width
    scale_tiles = (channels // 32 + 3) // 4
    padded_rows = (rows + 127) // 128 * 128
    quantized = torch.empty((rows, channels), device=value.device, dtype=torch.float8_e4m3fn)
    scale_bits = torch.zeros(padded_rows * scale_tiles * 4, device=value.device, dtype=torch.uint8)
    kernel = _kernels().depthwise5x5_relu_mxfp8_kernel(batch, height, width, channels)
    kernel(value.permute(0, 2, 3, 1), weight.reshape(-1), bias, quantized.view(-1), scale_bits)
    return quantized, scale_bits.view(torch.float8_e8m0fnu)


@_depthwise5x5_relu_mxfp8.register_fake
def _(value: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    batch, channels, height, width = value.shape
    rows = batch * height * width
    scale_tiles = (channels // 32 + 3) // 4
    padded_rows = (rows + 127) // 128 * 128
    return (
        value.new_empty((rows, channels), dtype=torch.float8_e4m3fn),
        value.new_empty((padded_rows * scale_tiles * 4,), dtype=torch.float8_e8m0fnu),
    )


def depthwise5x5_relu_mxfp8(value: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """``value`` channels-last BF16 ``[B, C, H, W]`` (C % 64 == 0), ``weight`` BF16 ``[25, C]``,
    ``bias`` FP32 ``[C]``. Returns E4M3 ``[B*H*W, C]`` and swizzled E8M0 scales."""
    return _depthwise5x5_relu_mxfp8(value, weight, bias)


@torch.library.custom_op("flashim::depthwise5x5_relu_int8_tilelang", mutates_args=())
def _depthwise5x5_relu_int8(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    if value.data_ptr() % 16:
        value = value.clone(memory_format=torch.channels_last)
    batch, channels, height, width = value.shape
    quantized = torch.empty((batch * height * width, channels), device=value.device, dtype=torch.int8)
    kernel = _kernels().depthwise5x5_relu_int8_kernel(batch, height, width, channels)
    kernel(value.permute(0, 2, 3, 1), weight.reshape(-1), bias, scale.reshape(1), quantized.view(-1))
    return quantized


@_depthwise5x5_relu_int8.register_fake
def _(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    batch, channels, height, width = value.shape
    return value.new_empty((batch * height * width, channels), dtype=torch.int8)


def depthwise5x5_relu_int8(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    """Same operands as the Triton kernel (``value`` channels-last BF16, ``C % 64 == 0``); int8 ``[B*H*W, C]``."""
    return _depthwise5x5_relu_int8(value, weight, bias, scale)


# -- registrations ---------------------------------------------------------------------------------------


def _eligible(value: Tensor, weight: Tensor, bias: Tensor, *_) -> bool:
    return (
        value.ndim == 4
        and value.shape[1] % 64 == 0
        # A single column block (width <= 16) miscompiles in TileLang 0.1.14.
        and value.shape[3] > 16
        and value.dtype == torch.bfloat16
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(weight.shape) == (25, value.shape[1])
        and weight.dtype == torch.bfloat16
        and bias.dtype == torch.float32
    )


_TILELANG = {"device": DeviceSpec("cuda"), "requires": [Feature.TILELANG, Feature.TMA], "priority": 1, "when": _eligible}
ops.depthwise5x5_relu_mxfp8.register("tilelang", **_TILELANG)(depthwise5x5_relu_mxfp8)
ops.depthwise5x5_relu_int8.register("tilelang", **_TILELANG)(depthwise5x5_relu_int8)
