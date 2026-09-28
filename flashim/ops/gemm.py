"""GEMMs: block-scaled MXFP8, static-activation FP8, and FP32 linear of low-precision inputs.

``linear_fp32``: the reference runs an FP32 GEMM; the ``triton`` path covers small outputs
(``N <= 16``) exactly; ``triton_split`` covers larger ones on tensor cores with the FP32 weight split
into two BF16 halves (~16 mantissa bits, not bit-exact).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op
from flashim.platforms import Feature
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight, quantize_static

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
    from flashim.ops._triton.gemm import fp8_gemm_bf16

    return fp8_gemm_bf16(activation, weight, None if bias is None else bias.float())


# -- static INT8 GEMM (per-channel weights, per-tensor activations) ---------------------------------


def _int8_linear_reference(a: Tensor, weight: Tensor, a_scale: Tensor, w_scale: Tensor, bias: Tensor | None) -> Tensor:
    """``[M, K] int8 x [N, K] int8^T`` in INT32, then ``acc * (a_scale * w_scale) + bias`` -> BF16."""
    acc = a.long() @ weight.long().t() if a.device.type == "cpu" else a.double() @ weight.double().t()
    out = acc.float() * (a_scale.float() * w_scale.float())
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


int8_linear = Op("int8_linear", _int8_linear_reference)


def _int_mm_eligible(a, weight, a_scale, w_scale, bias) -> bool:
    # torch._int_mm: M > 16, K and N multiples of 8 (cuBLASLt IMMA).
    return a.shape[0] > 16 and a.shape[1] % 8 == 0 and weight.shape[0] % 8 == 0 and a.is_contiguous()


@int8_linear.register("cublas", device=DeviceSpec("cuda", min_capability=(7, 5)), when=_int_mm_eligible)
def _int8_linear_cublas(a, weight, a_scale, w_scale, bias):
    out = torch._int_mm(a, weight.t()).float() * (a_scale * w_scale)
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


# The fused-epilogue Triton GEMM beats cuBLAS ``_int_mm`` + separate dequant passes at every shape.
@int8_linear.register(
    "triton", device=DeviceSpec("cuda", min_capability=(7, 5)), requires=[Feature.TRITON], priority=1
)
def _int8_linear_triton(a, weight, a_scale, w_scale, bias):
    from flashim.ops._triton.gemm import int8_gemm_bf16

    return int8_gemm_bf16(a.contiguous(), weight, a_scale, w_scale, bias)


# -- FP32 linear of a BF16 input ----------------------------------------------------------------


def _linear_fp32_reference(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """``F.linear`` in FP32 of a (typically BF16) input: ``[..., K] -> [..., N]``."""
    return F.linear(x.float(), weight, bias)


linear_fp32 = Op("linear_fp32", _linear_fp32_reference)


def _linear_fp32_eligible(x: Tensor, weight: Tensor, bias: Tensor) -> bool:
    k = x.shape[-1]
    return (
        x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.shape[0] <= 16
        and weight.shape[1] == k
        and k in (16, 32, 64, 128)
        and x.numel() < 2**31
    )


@linear_fp32.register("triton", device=DeviceSpec("cuda"), when=_linear_fp32_eligible)
def _linear_fp32_triton(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    from flashim.ops._triton.gemm import linear_fp32_small_n as kernel

    return kernel(x, weight, bias)


def _linear_fp32_split_eligible(x: Tensor, weight: Tensor, bias: Tensor) -> bool:
    return (
        x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.shape[0] > 16
        and weight.shape[1] == x.shape[-1]
        and x.shape[-1] % 32 == 0
        and x.numel() < 2**31
    )


@linear_fp32.register("triton_split", device=DeviceSpec("cuda", min_capability=(8, 0)), when=_linear_fp32_split_eligible)
def _linear_fp32_triton_split(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Tensor-core path: exact BF16 input, weight as two BF16 halves (~1e-5 relative error vs FP32)."""
    from flashim.ops._triton.gemm import linear_fp32_split as kernel

    return kernel(x, weight, bias)


# -- fused MXFP8 GELU MLP -------------------------------------------------------------------------------


def _mxfp8_gelu_mlp_reference(
    x: QuantizedTensor, w1: QuantizedWeight, b1: Tensor, hidden_exponent: Tensor, w2: QuantizedWeight
) -> Tensor:
    """``quantize_static(GELU(x W1^T [BF16] + b1) [BF16], hidden_exponent) W2^T`` -> BF16 ``[rows, N2]``
    (padded widths; W2's bias is not added). Same rounding points as two static MXFP8 linears around
    ``gelu_quantize(input_bias=b1)``."""
    hidden = _fp8_linear_reference(x, w1, None)
    activated = F.gelu(hidden.float() + b1.float()).to(torch.bfloat16)
    return _fp8_linear_reference(quantize_static(activated, hidden_exponent), w2, None)


mxfp8_gelu_mlp = Op("mxfp8_gelu_mlp", _mxfp8_gelu_mlp_reference, composite=True)


def _mlp_eligible(x, w1, b1, hidden_exponent, w2) -> bool:
    k1, hidden, n2 = x.channels, w1.out_features, w2.out_features
    return (
        w1.in_features == k1
        and w2.in_features == hidden
        and k1 in (32, 64, 128, 256)
        and n2 in (32, 64, 128, 256)
        and hidden % 64 == 0
    )


@mxfp8_gelu_mlp.register(
    "triton", device=DeviceSpec("cuda", min_capability=(10, 0)), requires=[Feature.TRITON], when=_mlp_eligible
)
def _mxfp8_gelu_mlp_triton(x, w1, b1, hidden_exponent, w2):
    from flashim.ops._triton.gemm import mxfp8_gelu_mlp as kernel

    return kernel(x, w1, b1, hidden_exponent.to(x.device, torch.int8), w2)
