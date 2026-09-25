"""MXFP8 ops: fused producers that emit quantized activations, and scaled GEMMs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op
from flashim.platforms import Feature
from flashim.quant.formats.mx import quantize_dynamic_2d, swizzle_scales
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight, quantize_static

_TRITON_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


def _cuda_row_major(value: Tensor) -> bool:
    return value.dtype in _TRITON_FLOATS and value.is_contiguous() and value.numel() > 0


# -- LayerNorm (+ residual) -> static quantize ------------------------------------------


def _layernorm_quantize_reference(
    value: Tensor,
    residual: Tensor | None,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
    scale_exponent: Tensor,
) -> QuantizedTensor:
    """``quantize_static(LayerNorm(value [+ residual]))`` with FP32 statistics.

    The normalized tensor is rounded to ``value.dtype`` before quantization.
    """
    dense = value.float() if residual is None else value.float() + residual.float()
    normalized = F.layer_norm(
        dense,
        (value.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).to(value.dtype)
    return quantize_static(normalized, scale_exponent)


layernorm_quantize = Op("layernorm_quantize", _layernorm_quantize_reference)


def _layernorm_quantize_eligible(value, residual, weight, bias, eps, scale_exponent) -> bool:
    return _cuda_row_major(value) and (
        residual is None or (residual.shape == value.shape and _cuda_row_major(residual))
    )


@layernorm_quantize.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_layernorm_quantize_eligible
)
def _layernorm_quantize_triton(value, residual, weight, bias, eps, scale_exponent):
    from flashim.ops._triton.mx import layernorm_quantize as kernel

    return kernel(value, residual, weight, bias, eps, scale_exponent.to(value.device, torch.int8))


# -- GELU -> static quantize ---------------------------------------------------------------


def _gelu_quantize_reference(value: Tensor, scale_exponent: Tensor, approximate: str = "none") -> QuantizedTensor:
    """``quantize_static(GELU(value))``; GELU is evaluated in ``value.dtype``."""
    return quantize_static(F.gelu(value, approximate=approximate), scale_exponent)


gelu_quantize = Op("gelu_quantize", _gelu_quantize_reference)


@gelu_quantize.register(
    "triton",
    device=DeviceSpec("cuda"),
    requires=[Feature.TRITON],
    when=lambda value, *args, **kwargs: _cuda_row_major(value),
)
def _gelu_quantize_triton(value: Tensor, scale_exponent: Tensor, approximate: str = "none"):
    from flashim.ops._triton.mx import gelu_quantize as kernel

    return kernel(value, scale_exponent.to(value.device, torch.int8), approximate)


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
    from flashim.ops._triton.mx import depthwise5x5_relu_mxfp8 as kernel

    return kernel(value, weight, bias)


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
    from flashim.ops._triton.mx import depthwise5x5_relu_pointwise as kernel

    return kernel(value, dw_weight, dw_bias, pw_weight, pw_bias)


# -- whole MXFP8 refiner block: depthwise 5x5 + ReLU -> dynamic MXFP8 -> pointwise GEMM -----------


def _mxfp8_refiner_block_reference(
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


mxfp8_refiner_block = Op("mxfp8_refiner_block", _mxfp8_refiner_block_reference)


def _refiner_block_eligible(value, dw_weight, dw_bias, pw_weight, pw_scale_swizzled, pw_scale_u8, pw_bias) -> bool:
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


@mxfp8_refiner_block.register(
    "triton_fused", device=DeviceSpec("cuda", min_capability=(10, 0)), requires=[Feature.TRITON],
    when=_refiner_block_eligible,
)
def _mxfp8_refiner_block_triton(value, dw_weight, dw_bias, pw_weight, pw_scale_swizzled, pw_scale_u8, pw_bias):
    from flashim.ops._triton.mx import dw5x5_relu_mx_pointwise as kernel

    return kernel(value, dw_weight, dw_bias, pw_weight, pw_scale_u8, pw_bias)


# -- block-scaled MXFP8 GEMM -----------------------------------------------------------------


def _dequantize_swizzled(values: Tensor, swizzled: Tensor) -> Tensor:
    """Invert :func:`swizzle_scales` and dequantize a ``[rows, K]`` E4M3 matrix."""
    rows, k = values.shape
    groups = k // 32
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((groups + 3) // 4) * 4
    scales = (
        swizzled.view(torch.uint8)
        .reshape(-1, 32, 4, 4)
        .transpose(1, 2)
        .reshape(padded_rows // 128, padded_cols // 4, 128, 4)
        .permute(0, 2, 1, 3)
        .reshape(padded_rows, padded_cols)[:rows, :groups]
    )
    exponent = scales.to(torch.int16).float() - 127.0
    blocks = values.float().reshape(rows, groups, 32) * torch.exp2(exponent)[..., None]
    return blocks.reshape(rows, k)


def _mxfp8_scaled_mm_reference(
    a: Tensor, b_t: Tensor, a_scale: Tensor, b_scale: Tensor, bias: Tensor | None = None
) -> Tensor:
    """Emulated ``F.scaled_mm`` MXFP8: ``a [M,K]`` x ``b_t [K,N]`` with swizzled scales, BF16 out."""
    a_dense = _dequantize_swizzled(a, a_scale)
    b_dense = _dequantize_swizzled(b_t.t(), b_scale)
    out = a_dense @ b_dense.t()
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


mxfp8_scaled_mm = Op("mxfp8_scaled_mm", _mxfp8_scaled_mm_reference)


@torch.compiler.allow_in_graph
def _scaled_mm_cublas(a: Tensor, b_t: Tensor, a_scale: Tensor, b_scale: Tensor, bias: Tensor | None) -> Tensor:
    # The ScalingType/SwizzleType pybind enums cannot be proxied by Dynamo, so they
    # stay inside this allow_in_graph boundary.
    return F.scaled_mm(
        a,
        b_t,
        a_scale,
        F.ScalingType.BlockWise1x32,
        b_scale,
        F.ScalingType.BlockWise1x32,
        F.SwizzleType.SWIZZLE_32_4_4,
        F.SwizzleType.SWIZZLE_32_4_4,
        bias=bias,
        output_dtype=torch.bfloat16,
    )


@mxfp8_scaled_mm.register("cublas", device=DeviceSpec("cuda"), requires=[Feature.MXFP8_SCALED_MM])
def _mxfp8_scaled_mm_cublas(a, b_t, a_scale, b_scale, bias=None):
    return _scaled_mm_cublas(a, b_t, a_scale, b_scale, bias)


# -- static-activation FP8 GEMM for weights cuBLAS cannot take (K or N % 128 != 0) --------------


def _fp8_linear_reference(activation: QuantizedTensor, weight: QuantizedWeight, bias: Tensor | None) -> Tensor:
    """Dequantize both operands and matmul in FP32; padded ``[rows, N]`` BF16 output."""
    a = activation.matrix().float() * torch.exp2(activation.scale_exponent.float()).repeat_interleave(32)
    w = weight.values.float() * torch.exp2(weight.scale_exponent.float()).repeat_interleave(32, dim=1)
    out = a @ w.t()
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


fp8_linear = Op("fp8_linear", _fp8_linear_reference)


@fp8_linear.register("triton", device=DeviceSpec("cuda", min_capability=(8, 9)), requires=[Feature.TRITON])
def _fp8_linear_triton(activation: QuantizedTensor, weight: QuantizedWeight, bias: Tensor | None) -> Tensor:
    from flashim.ops._triton.mx import fp8_gemm_bf16

    return fp8_gemm_bf16(activation, weight, None if bias is None else bias.float())
