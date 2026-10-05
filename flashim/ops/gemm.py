"""GEMMs: block-scaled MXFP8, static MXFP8 / INT8 linears, and FP32 linear of low-precision inputs.

``linear_fp32``: the reference runs an FP32 GEMM; the ``triton`` path covers small outputs
(``N <= 16``) exactly; ``triton_split`` covers larger ones on tensor cores with the FP32 weight split
into two BF16 halves (~16 mantissa bits, not bit-exact).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
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


# -- static MXFP8 linear ---------------------------------------------------------------------


def _mxfp8_linear_reference(activation: QuantizedTensor, bias: Tensor | None, weight: QuantizedWeight) -> Tensor:
    """Static MXFP8 linear, emulated: dequantize both operands and matmul in FP32; padded ``[rows, N]`` BF16 out.

    ``activation`` block-32 E4M3 with its (static) E8M0 exponents; ``weight`` E4M3 ``[N, K]`` with one exponent per
    (row, 32 inputs), both padded to multiples of 32; ``bias`` (padded ``[N]``, optional) is added in the FP32
    epilogue. ``weight`` is the bound operand: impls repack it once (``Op.bind``)."""
    a = activation.matrix().float() * torch.exp2(activation.scale_exponent.float()).repeat_interleave(32)
    w = weight.values.float() * torch.exp2(weight.scale_exponent.float()).repeat_interleave(32, dim=1)
    out = a @ w.t()
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


mxfp8_linear = Op("mxfp8_linear", _mxfp8_linear_reference, weights=1)


# -- static INT8 GEMM (per-channel weights, per-tensor activations) ---------------------------------


def _int8_linear_reference(
    a: Tensor, weight: Tensor, a_scale: Tensor, w_scale: Tensor, bias: Tensor | None, *, relu: bool = False
) -> Tensor:
    """``[M, K] int8 x [N, K] int8^T`` in INT32, then ``acc * (a_scale * w_scale) + bias`` (``relu``: then ReLU) -> BF16.
    ReLU commutes with the BF16 rounding, so ``relu=True`` equals ``F.relu`` of the ``relu=False`` result bit for bit."""
    acc = a.long() @ weight.long().t() if a.device.type == "cpu" else a.double() @ weight.double().t()
    out = acc.float() * (a_scale.float() * w_scale.float())
    if bias is not None:
        out = out + bias.float()
    if relu:
        out = out.relu()
    return out.to(torch.bfloat16)


int8_linear = Op("int8_linear", _int8_linear_reference)


# -- FP32 linear of a BF16 input ----------------------------------------------------------------


def _linear_fp32_reference(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """``F.linear`` in FP32 of a (typically BF16) input: ``[..., K] -> [..., N]``."""
    return F.linear(x.float(), weight, bias)


linear_fp32 = Op("linear_fp32", _linear_fp32_reference)


# -- fused MXFP8 GELU MLP -------------------------------------------------------------------------------


def _mxfp8_gelu_mlp_reference(
    x: QuantizedTensor, w1: QuantizedWeight, b1: Tensor, hidden_exponent: Tensor, w2: QuantizedWeight
) -> Tensor:
    """``quantize_static(GELU(x W1^T [BF16] + b1) [BF16], hidden_exponent) W2^T`` -> BF16 ``[rows, N2]``
    (padded widths; W2's bias is not added). Same rounding points as two static MXFP8 linears around
    ``gelu_quantize(input_bias=b1)``."""
    hidden = _mxfp8_linear_reference(x, None, w1)
    activated = F.gelu(hidden.float() + b1.float()).to(torch.bfloat16)
    return _mxfp8_linear_reference(quantize_static(activated, hidden_exponent), None, w2)


mxfp8_gelu_mlp = Op("mxfp8_gelu_mlp", _mxfp8_gelu_mlp_reference)
