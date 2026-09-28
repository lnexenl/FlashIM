"""Depthwise-separable conv blocks: depthwise 5x5 + bias + ReLU (+ pointwise 1x1), BF16 or
with dynamic MXFP8 activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op
from flashim.ops.gemm import int8_linear, mxfp8_scaled_mm
from flashim.platforms import Feature
from flashim.quant.formats.int8 import quantize_symmetric
from flashim.quant.formats.mx import quantize_dynamic_2d, swizzle_scales

# -- depthwise 5x5 + bias + ReLU -> dynamic MXFP8 ----------------------------------------------


def _depthwise5x5_relu_mxfp8_reference(value: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """Depthwise 5x5 (pad 2) on BF16-rounded inputs, + bias, ReLU, dynamic MXFP8.

    ``value`` is ``[B, C, H, W]``; ``weight`` is packed ``[25, C]`` (ky, kx major);
    ``bias`` is ``[C]``. Returns E4M3 ``[B*H*W, C]`` and swizzled E8M0 scales.
    """
    channels = value.shape[1]
    kernel = weight.float().t().reshape(channels, 1, 5, 5)
    rounded = value.to(torch.bfloat16).float()
    out = F.relu(F.conv2d(rounded, kernel, bias.float(), padding=2, groups=channels))
    quantized, exponent = quantize_dynamic_2d(out.permute(0, 2, 3, 1).reshape(-1, channels))
    return quantized, swizzle_scales(exponent)


depthwise5x5_relu_mxfp8 = Op("depthwise5x5_relu_mxfp8", _depthwise5x5_relu_mxfp8_reference)


def _depthwise_eligible(value: Tensor, weight: Tensor, bias: Tensor) -> bool:
    return (
        value.ndim == 4
        and value.shape[1] in (32, 128, 512)
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(weight.shape) == (25, value.shape[1])
    )


@depthwise5x5_relu_mxfp8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_depthwise_eligible
)
def _depthwise5x5_relu_mxfp8_triton(value: Tensor, weight: Tensor, bias: Tensor):
    from flashim.ops._triton.conv import depthwise5x5_relu_mxfp8 as kernel

    return kernel(value, weight, bias)


def _depthwise_tilelang_eligible(value: Tensor, weight: Tensor, bias: Tensor) -> bool:
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


@depthwise5x5_relu_mxfp8.register(
    "tilelang",
    device=DeviceSpec("cuda"),
    requires=[Feature.TILELANG, Feature.TMA],
    priority=1,
    when=_depthwise_tilelang_eligible,
)
def _depthwise5x5_relu_mxfp8_tilelang(value: Tensor, weight: Tensor, bias: Tensor):
    from flashim.ops._tilelang.conv import depthwise5x5_relu_mxfp8 as kernel

    return kernel(value, weight, bias)


# -- depthwise 5x5 + bias + ReLU -> static INT8 ---------------------------------------------------


def _depthwise5x5_relu_int8_reference(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    """Depthwise 5x5 (pad 2) on BF16-rounded inputs, + bias, ReLU, rounded to BF16, then static INT8.

    Same operands as :func:`depthwise5x5_relu_mxfp8`; returns int8 ``[B*H*W, C]`` (``scale`` FP32).
    """
    channels = value.shape[1]
    kernel = weight.float().t().reshape(channels, 1, 5, 5)
    out = F.relu(F.conv2d(value.to(torch.bfloat16).float(), kernel, bias.float(), padding=2, groups=channels))
    return quantize_symmetric(out.to(torch.bfloat16).permute(0, 2, 3, 1).reshape(-1, channels), scale)


depthwise5x5_relu_int8 = Op("depthwise5x5_relu_int8", _depthwise5x5_relu_int8_reference)


@depthwise5x5_relu_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=lambda value, weight, bias, scale: _depthwise_eligible(value, weight, bias)
)
def _depthwise5x5_relu_int8_triton(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor):
    from flashim.ops._triton.conv import depthwise5x5_relu_int8 as kernel

    return kernel(value, weight, bias, scale)


@depthwise5x5_relu_int8.register(
    "tilelang",
    device=DeviceSpec("cuda"),
    requires=[Feature.TILELANG, Feature.TMA],
    priority=1,
    when=lambda value, weight, bias, scale: _depthwise_tilelang_eligible(value, weight, bias),
)
def _depthwise5x5_relu_int8_tilelang(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor):
    from flashim.ops._tilelang.conv import depthwise5x5_relu_int8 as kernel

    return kernel(value, weight, bias, scale)


# -- BF16 depthwise 5x5 + bias + ReLU + pointwise + bias (fused refiner block) -----------------


def _depthwise5x5_relu_pointwise_reference(
    value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor
) -> Tensor:
    """BF16 refiner conv block with FP32 accumulation.

    ``value`` ``[B, C, H, W]`` (rounded to BF16 on load); ``dw_weight`` packed ``[25, C]``;
    ``pw_weight`` ``[C_out, C_in]``; biases ``[C]``. The depthwise result is rounded to
    BF16 before the pointwise product. Returns channels-last BF16 ``[B, C, H, W]``.
    """
    channels = value.shape[1]
    kernel = dw_weight.float().t().reshape(channels, 1, 5, 5)
    hidden = F.conv2d(value.to(torch.bfloat16).float(), kernel, dw_bias.float(), padding=2, groups=channels)
    hidden = F.relu(hidden).to(torch.bfloat16).float()
    out = F.conv2d(hidden, pw_weight.float()[:, :, None, None], pw_bias.float())
    return out.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)


depthwise5x5_relu_pointwise = Op("depthwise5x5_relu_pointwise", _depthwise5x5_relu_pointwise_reference)


def _dw_pw_eligible(value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor) -> bool:
    channels = value.shape[1] if value.ndim == 4 else 0
    return (
        channels in (16, 32, 64)
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(dw_weight.shape) == (25, channels)
        and tuple(pw_weight.shape) == (channels, channels)
        and pw_weight.dtype == torch.bfloat16
        and pw_weight.is_contiguous()
        and (value.numel() + 128 * channels) < 2**31
    )


@depthwise5x5_relu_pointwise.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_dw_pw_eligible
)
def _depthwise5x5_relu_pointwise_triton(value, dw_weight, dw_bias, pw_weight, pw_bias):
    from flashim.ops._triton.conv import depthwise5x5_relu_pointwise as kernel

    return kernel(value, dw_weight, dw_bias, pw_weight, pw_bias)


# -- MXFP8 depthwise-separable block: depthwise 5x5 + ReLU -> dynamic MXFP8 -> pointwise GEMM -----------


def _depthwise5x5_relu_pointwise_mxfp8_reference(
    value: Tensor,
    dw_weight: Tensor,
    dw_bias: Tensor,
    pw_weight: Tensor,
    pw_scale_swizzled: Tensor,
    pw_scale_u8: Tensor,
    pw_bias: Tensor,
) -> Tensor:
    """The unfused pipeline through normal op dispatch: ``depthwise5x5_relu_mxfp8`` then
    ``mxfp8_scaled_mm``. ``pw_weight`` ``[N, K]`` E4M3; ``pw_scale_swizzled`` is the cuBLAS layout
    of the weight exponents and ``pw_scale_u8`` the plain biased ``[N, K/32]`` bytes."""
    batch, channels, height, width = value.shape
    quantized, scale = depthwise5x5_relu_mxfp8(value, dw_weight, dw_bias)
    out = mxfp8_scaled_mm(quantized, pw_weight.t(), scale, pw_scale_swizzled, pw_bias)
    return out.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


depthwise5x5_relu_pointwise_mxfp8 = Op(
    "depthwise5x5_relu_pointwise_mxfp8", _depthwise5x5_relu_pointwise_mxfp8_reference, composite=True
)


def _mxfp8_block_eligible(value, dw_weight, dw_bias, pw_weight, pw_scale_swizzled, pw_scale_u8, pw_bias) -> bool:
    return (
        value.ndim == 4
        and value.shape[1] == 128
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(dw_weight.shape) == (25, 128)
        and tuple(pw_weight.shape) == (128, 128)
        and pw_weight.dtype == torch.float8_e4m3fn
        and pw_scale_u8.dtype == torch.uint8
        and tuple(pw_scale_u8.shape) == (128, 4)
    )


@depthwise5x5_relu_pointwise_mxfp8.register(
    "triton_fused", device=DeviceSpec("cuda", min_capability=(10, 0)), requires=[Feature.TRITON],
    when=_mxfp8_block_eligible,
)
def _depthwise5x5_relu_pointwise_mxfp8_triton(value, dw_weight, dw_bias, pw_weight, pw_scale_swizzled, pw_scale_u8, pw_bias):
    from flashim.ops._triton.conv import depthwise5x5_relu_pointwise_mxfp8 as kernel

    return kernel(value, dw_weight, dw_bias, pw_weight, pw_scale_u8, pw_bias)


# -- INT8 depthwise-separable block: depthwise 5x5 + ReLU -> static INT8 -> pointwise GEMM ------------


def _depthwise5x5_relu_pointwise_int8_reference(
    value: Tensor,
    dw_weight: Tensor,
    dw_bias: Tensor,
    pw_weight: Tensor,
    pw_wscale: Tensor,
    pw_bias: Tensor,
    scale: Tensor,
) -> Tensor:
    """The unfused pipeline through normal op dispatch: ``depthwise5x5_relu_int8`` then ``int8_linear``.
    ``pw_weight`` int8 ``[N, K]`` with per-row FP32 ``pw_wscale``; ``scale`` the static input scale."""
    batch, channels, height, width = value.shape
    quantized = depthwise5x5_relu_int8(value, dw_weight, dw_bias, scale)
    out = int8_linear(quantized, pw_weight, scale, pw_wscale, pw_bias)
    return out.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


depthwise5x5_relu_pointwise_int8 = Op(
    "depthwise5x5_relu_pointwise_int8", _depthwise5x5_relu_pointwise_int8_reference, composite=True
)


def _int8_block_eligible(value, dw_weight, dw_bias, pw_weight, pw_wscale, pw_bias, scale) -> bool:
    return (
        value.ndim == 4
        and value.shape[1] == 128
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(dw_weight.shape) == (25, 128)
        and tuple(pw_weight.shape) == (128, 128)
        and pw_weight.dtype == torch.int8
    )


@depthwise5x5_relu_pointwise_int8.register(
    "triton_fused", device=DeviceSpec("cuda", min_capability=(8, 0)), requires=[Feature.TRITON], when=_int8_block_eligible
)
def _depthwise5x5_relu_pointwise_int8_triton(value, dw_weight, dw_bias, pw_weight, pw_wscale, pw_bias, scale):
    from flashim.ops._triton.conv import depthwise5x5_relu_pointwise_int8 as kernel

    return kernel(value, dw_weight, dw_bias, pw_weight, pw_wscale, pw_bias, scale)
