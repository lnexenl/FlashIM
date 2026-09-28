"""Depthwise-separable conv blocks: depthwise 5x5 + bias + ReLU (+ pointwise 1x1), BF16 or
with dynamic MXFP8 activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.ops.gemm import int8_linear, mxfp8_scaled_mm
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
