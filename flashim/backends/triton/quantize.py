# Portions adapted from RoMaV2 (MIT, Copyright (c) 2025 Johan Edstedt); see THIRD_PARTY_NOTICES.md.
"""Producer -> static MXFP8 quantize kernels: LayerNorm (+ residual) and GELU (ported from
RoMaV2 ``fast_fp8_fusion.py`` / ``full_fp8.py``; byte-identical numerics).

Norm/activation math in FP32, rounded to the dense dtype before quantization; static
quantization saturates to +-448 (E4M3).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature
from flashim.quant.formats.mx import padded_channels
from flashim.quant.tensor import Int8Tensor, QuantizedTensor

_ROUND_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


@triton.jit
def _round_to_dense(value, ROUND: tl.constexpr):
    if ROUND == 1:
        return value.to(tl.bfloat16).to(tl.float32)
    if ROUND == 2:
        return value.to(tl.float16).to(tl.float32)
    return value.to(tl.float32)


def _f32_to_e4m3_double_rounds() -> bool:
    return torch.cuda.is_available() and any(
        torch.cuda.get_device_capability(index) < (9, 0) for index in range(torch.cuda.device_count())
    )


# Triton 3.6 emits ``cvt.rn.satfinite.e4m3x2.f32`` only for sm_90+. Below, FP32 -> E4M3 is lowered as
# ``cvt.rz.f16.f32`` + ``cvt.rn.satfinite.e4m3x2.f16x2``: truncating first sends 0.4% of FP32 values to the wrong
# neighbour (measured on an L20 against ``Tensor.to(float8_e4m3fn)``; BF16-rounded values fit FP16 and are exact).
# sm_89 has the FP32 instruction, so it is emitted by hand whenever such a device is present (byte-identical to
# torch on sm_89, signed zeros / subnormals / saturation included; producers measured within 3% of the FP16 route
# there); sm_90+ alone keeps Triton's own lowering.
_E4M3_INLINE_ASM = tl.constexpr(_f32_to_e4m3_double_rounds())


@triton.jit
def to_e4m3(value):
    """FP32 -> E4M3, round to nearest even, saturating: ``value.to(tl.float8e4nv)`` with one rounding on sm_89."""
    if _E4M3_INLINE_ASM:  # pack=1: any block shape (pack=2 does not compile for single-element blocks)
        code = tl.inline_asm_elementwise(
            "{ .reg .b16 pair; cvt.rn.satfinite.e4m3x2.f32 pair, $1, $1; cvt.u32.u16 $0, pair; }",
            "=r,r", [value], dtype=tl.int8, is_pure=True, pack=1,
        )
        return code.to(tl.float8e4nv, bitcast=True)
    return value.to(tl.float8e4nv)


@triton.jit
def _store_quantized(out_ptr, row, cols, activated, scale_ptr, padded_channels, valid=True, INT8: tl.constexpr = False):
    store = (cols < padded_channels) & valid
    if INT8:  # one static FP32 scale; round half to even, saturate to +-127
        scale = tl.load(scale_ptr)
        scaled = libdevice.rint(activated * (1.0 / scale))  # reciprocal multiply: exact division is ~2x slower
        tl.store(out_ptr + row * padded_channels + cols, tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8), mask=store)
    else:
        exponent = tl.load(scale_ptr + cols // 32, mask=store, other=0).to(tl.float32)
        scaled = activated * tl.exp2(-exponent)
        quantized = to_e4m3(tl.maximum(tl.minimum(scaled, 448.0), -448.0))
        tl.store(out_ptr + row * padded_channels + cols, quantized, mask=store)


@triton.jit
def _gelu(value, TANH: tl.constexpr):
    if TANH:
        inner = 0.7978845608028654 * (value + 0.044715 * value * value * value)
        return 0.5 * value * (1.0 + (1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)))
    return 0.5 * value * (1.0 + tl.math.erf(value * 0.7071067811865476))


@triton.jit
def _gelu_quantize_kernel(
    value_ptr,
    scale_ptr,
    out_ptr,
    input_bias_ptr,
    channels: tl.constexpr,
    padded_channels: tl.constexpr,
    elements,
    BLOCK: tl.constexpr,
    TANH: tl.constexpr,
    ROUND: tl.constexpr,
    INT8: tl.constexpr,
    HAS_INPUT_BIAS: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // padded_channels
    cols = offsets % padded_channels
    valid = offsets < elements
    value = tl.load(value_ptr + row * channels + cols, mask=valid & (cols < channels), other=0.0).to(
        tl.float32
    )
    if HAS_INPUT_BIAS:
        value += tl.load(input_bias_ptr + cols, mask=valid & (cols < channels), other=0.0).to(tl.float32)
    if TANH:
        inner = 0.7978845608028654 * (value + 0.044715 * value * value * value)
        tanh = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
        activated = 0.5 * value * (1.0 + tanh)
    else:
        activated = 0.5 * value * (1.0 + tl.math.erf(value * 0.7071067811865476))
    activated = _round_to_dense(activated, ROUND)
    _store_quantized(out_ptr, row, cols, activated, scale_ptr, padded_channels, valid, INT8)


@triton.jit
def _layernorm_quantize_kernel(
    value_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    out_ptr,
    input_bias_ptr,
    dense_ptr,
    sum_ptr,
    channels,
    padded_channels,
    eps,
    BLOCK_C: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROUND: tl.constexpr,
    INT8: tl.constexpr,
    HAS_INPUT_BIAS: tl.constexpr,
    DENSE_OUT: tl.constexpr,
    SUM_OUT: tl.constexpr,
    GELU: tl.constexpr,
    TANH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    value_base = row * channels
    partial = tl.zeros([BLOCK_C], dtype=tl.float32)
    partial_sq = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c0 in range(0, channels, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        mask = cols < channels
        value = tl.load(value_ptr + value_base + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_INPUT_BIAS:
            value += tl.load(input_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + value_base + cols, mask=mask, other=0.0).to(tl.float32)
            value = value + residual
        partial += tl.where(mask, value, 0.0)
        partial_sq += tl.where(mask, value * value, 0.0)
    mean = tl.sum(partial) / channels
    variance = tl.maximum(tl.sum(partial_sq) / channels - mean * mean, 0.0)
    rstd = tl.rsqrt(variance + tl.cast(eps, tl.float32))
    for c0 in range(0, padded_channels, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        valid = cols < channels
        value = tl.load(value_ptr + value_base + cols, mask=valid, other=0.0).to(tl.float32)
        if HAS_INPUT_BIAS:
            value += tl.load(input_bias_ptr + cols, mask=valid, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + value_base + cols, mask=valid, other=0.0).to(tl.float32)
            value = value + residual
        if SUM_OUT:  # the FP32 sum rounded once, as a BF16 torch.add would
            tl.store(sum_ptr + value_base + cols, value.to(sum_ptr.dtype.element_ty), mask=valid)
        out = (value - mean) * rstd
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + cols, mask=valid, other=0.0).to(tl.float32)
            out = out * weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=valid, other=0.0).to(tl.float32)
            out = out + bias
        out = tl.where(valid, out, 0.0)
        out = _round_to_dense(out, ROUND)
        if GELU:  # on the rounded normalized value, rounded again (nn.LayerNorm -> nn.GELU in the dense dtype)
            out = _round_to_dense(_gelu(out, TANH), ROUND)
        if DENSE_OUT:
            tl.store(dense_ptr + value_base + cols, out.to(dense_ptr.dtype.element_ty), mask=valid)
        _store_quantized(out_ptr, row, cols, out, scale_ptr, padded_channels, True, INT8)


@triton.jit
def _quantize_static_kernel(
    value_ptr,
    scale_ptr,
    out_ptr,
    channels: tl.constexpr,
    padded_channels: tl.constexpr,
    elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // padded_channels
    cols = offsets % padded_channels
    valid = offsets < elements
    value = tl.load(value_ptr + row * channels + cols, mask=valid & (cols < channels), other=0.0).to(tl.float32)
    _store_quantized(out_ptr, row, cols, value, scale_ptr, padded_channels, valid, False)


@triton.jit
def _quantize_channels_first_kernel(
    value_ptr,
    scale_ptr,
    out_ptr,
    channels,
    padded_channels,
    pixels,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INT8: tl.constexpr,
):
    """Static quantize of the ``[B, H, W, C]`` channels-last *view* of an NCHW-contiguous tensor
    (``x.permute(0, 2, 3, 1)``) into the packed ``[B, H, W, Cpad]`` layout. A tile is loaded along pixels (contiguous in
    NCHW) and stored along channels (contiguous in the output), so the layout change costs no separate copy pass; the
    per-element math is the row-major kernels' (bit-identical)."""
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    b = tl.program_id(2).to(tl.int64)
    p_ok = p < pixels
    load_ok = p_ok[:, None] & (c[None, :] < channels)
    value = tl.load(value_ptr + (b * channels + c[None, :]) * pixels + p[:, None], mask=load_ok, other=0.0)
    row = (b * pixels + p)[:, None]
    _store_quantized(out_ptr, row, c[None, :], value.to(tl.float32), scale_ptr, padded_channels, p_ok[:, None], INT8)


def _channels_first_view(value: Tensor) -> bool:
    """``value`` is ``x.permute(0, 2, 3, 1)`` of an NCHW-contiguous ``x`` (and not itself row-major)."""
    return value.ndim == 4 and not value.is_contiguous() and value.permute(0, 3, 1, 2).is_contiguous()


def _quantize_channels_first(value: Tensor, scale: Tensor, out: Tensor, padded: int, int8: bool) -> None:
    batch, height, width, channels = value.shape
    pixels = height * width
    block_p, block_c = 64, 64
    grid = (triton.cdiv(pixels, block_p), triton.cdiv(padded, block_c), batch)
    _quantize_channels_first_kernel[grid](
        value, scale, out, channels, padded, pixels, BLOCK_P=block_p, BLOCK_C=block_c, INT8=int8, num_warps=4
    )


def _empty_quantized(value: Tensor, padded: int) -> Tensor:
    return torch.empty((*value.shape[:-1], padded), device=value.device, dtype=torch.float8_e4m3fn)


def layernorm_quantize(
    value: Tensor,
    residual: Tensor | None,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
    scale_exponent: Tensor,
    *,
    input_bias: Tensor | None = None,
    return_dense: bool = False,
    return_sum: bool = False,
    activation: str | None = None,
) -> QuantizedTensor | tuple[QuantizedTensor, Tensor]:
    if return_dense and return_sum:
        raise ValueError("return_dense and return_sum are exclusive")
    channels = value.shape[-1]
    padded = padded_channels(channels)
    out = _empty_quantized(value, padded)
    dense = torch.empty_like(value) if return_dense else value
    summed = torch.empty_like(value) if return_sum else value
    rows = value.numel() // channels
    block_c = min(1024, triton.next_power_of_2(channels))
    _layernorm_quantize_kernel[(rows,)](
        value,
        residual if residual is not None else value,
        weight.contiguous().reshape(-1) if weight is not None else value,
        bias.contiguous().reshape(-1) if bias is not None else value,
        scale_exponent,
        out,
        input_bias if input_bias is not None else value,
        dense,
        summed,
        channels,
        padded,
        float(eps),
        BLOCK_C=block_c,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        ROUND=_ROUND_CODE[value.dtype],
        INT8=False,
        HAS_INPUT_BIAS=input_bias is not None,
        DENSE_OUT=return_dense,
        SUM_OUT=return_sum,
        GELU=activation is not None,
        TANH=activation == "tanh",
        num_warps=4 if block_c >= 128 else 1,
    )
    quantized = QuantizedTensor(out, scale_exponent, tuple(value.shape))
    if return_sum:
        return quantized, summed
    return (quantized, dense) if return_dense else quantized


def gelu_quantize(
    value: Tensor, scale_exponent: Tensor, approximate: str, *, input_bias: Tensor | None = None
) -> QuantizedTensor:
    channels = value.shape[-1]
    padded = padded_channels(channels)
    out = _empty_quantized(value, padded)
    rows = value.numel() // channels
    block = 1024
    _gelu_quantize_kernel[(triton.cdiv(rows * padded, block),)](
        value,
        scale_exponent,
        out,
        input_bias if input_bias is not None else value,
        channels,
        padded,
        rows * padded,
        BLOCK=block,
        TANH=approximate == "tanh",
        ROUND=_ROUND_CODE[value.dtype],
        INT8=False,
        HAS_INPUT_BIAS=input_bias is not None,
        num_warps=4,
    )
    return QuantizedTensor(out, scale_exponent, tuple(value.shape))


def quantize_static(value: Tensor, scale_exponent: Tensor) -> QuantizedTensor:
    channels = value.shape[-1]
    padded = padded_channels(channels)
    out = _empty_quantized(value, padded)
    if _channels_first_view(value):
        _quantize_channels_first(value, scale_exponent, out, padded, int8=False)
        return QuantizedTensor(out, scale_exponent, tuple(value.shape))
    rows = value.numel() // channels
    block = 1024
    _quantize_static_kernel[(triton.cdiv(rows * padded, block),)](
        value, scale_exponent, out, channels, padded, rows * padded, BLOCK=block, num_warps=4
    )
    return QuantizedTensor(out, scale_exponent, tuple(value.shape))


# -- INT8 (static per-tensor FP32 scale; no channel padding) ----------------------------------------


@triton.jit
def _quantize_int8_kernel(value_ptr, scale_ptr, out_ptr, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    value = tl.load(value_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    _store_quantized(out_ptr, 0, offsets, value, scale_ptr, elements, valid, True)


def quantize_int8(value: Tensor, scale: Tensor) -> Int8Tensor:
    out = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    if _channels_first_view(value):
        _quantize_channels_first(value, scale, out, value.shape[-1], int8=True)
        return Int8Tensor(out, scale)
    _quantize_int8_kernel[(triton.cdiv(value.numel(), 4096),)](value, scale, out, value.numel(), BLOCK=4096, num_warps=4)
    return Int8Tensor(out, scale)


def layernorm_quantize_int8(
    value: Tensor, residual: Tensor | None, weight: Tensor | None, bias: Tensor | None, eps: float, scale: Tensor,
    *, return_sum: bool = False, activation: str | None = None,
) -> Int8Tensor | tuple[Int8Tensor, Tensor]:
    channels = value.shape[-1]
    out = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    summed = torch.empty_like(value) if return_sum else value
    block_c = min(1024, triton.next_power_of_2(channels))
    _layernorm_quantize_kernel[(value.numel() // channels,)](
        value,
        residual if residual is not None else value,
        weight.contiguous().reshape(-1) if weight is not None else value,
        bias.contiguous().reshape(-1) if bias is not None else value,
        scale,
        out,
        value,
        value,
        summed,
        channels,
        channels,
        float(eps),
        BLOCK_C=block_c,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        ROUND=_ROUND_CODE[value.dtype],
        INT8=True,
        HAS_INPUT_BIAS=False,
        DENSE_OUT=False,
        SUM_OUT=return_sum,
        GELU=activation is not None,
        TANH=activation == "tanh",
        num_warps=4 if block_c >= 128 else 1,
    )
    return (Int8Tensor(out, scale), summed) if return_sum else Int8Tensor(out, scale)


def gelu_quantize_int8(value: Tensor, scale: Tensor, approximate: str) -> Int8Tensor:
    channels = value.shape[-1]
    out = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    _gelu_quantize_kernel[(triton.cdiv(value.numel(), 1024),)](
        value,
        scale,
        out,
        value,
        channels,
        channels,
        value.numel(),
        BLOCK=1024,
        TANH=approximate == "tanh",
        ROUND=_ROUND_CODE[value.dtype],
        INT8=True,
        HAS_INPUT_BIAS=False,
        num_warps=4,
    )
    return Int8Tensor(out, scale)

# -- registrations ---------------------------------------------------------------------------------------

def _row_major(value: Tensor) -> bool:
    return value.dtype in (torch.float16, torch.bfloat16, torch.float32) and value.is_contiguous() and value.numel() > 0


def _quantize_eligible(value: Tensor, *_) -> bool:
    """Row-major, or the channels-last view of NCHW memory (quantized convolutions' input: no copy first)."""
    floating = value.dtype in (torch.float16, torch.bfloat16, torch.float32)
    return _row_major(value) or (floating and value.numel() > 0 and _channels_first_view(value))


def _layernorm_eligible(value, residual, weight, bias, eps, scale_exponent, **kwargs) -> bool:
    return _row_major(value) and (residual is None or (residual.shape == value.shape and _row_major(residual)))


# The E4M3 producers need FP8 conversion instructions: Triton rejects ``tl.float8e4nv`` below sm_89 (compile-checked
# for sm_87, Jetson Orin), where the torch references run instead.
@ops.layernorm_quantize.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3], when=_layernorm_eligible
)
def _layernorm_quantize_impl(value, residual, weight, bias, eps, scale_exponent, **kwargs):
    return layernorm_quantize(value, residual, weight, bias, eps, scale_exponent.to(value.device, torch.int8), **kwargs)


@ops.gelu_quantize.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3],
    when=lambda value, *a, **k: _row_major(value),
)
def _gelu_quantize_impl(value: Tensor, scale_exponent: Tensor, approximate: str = "none", *, input_bias=None):
    return gelu_quantize(value, scale_exponent.to(value.device, torch.int8), approximate, input_bias=input_bias)


@ops.quantize_mxfp8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3], when=_quantize_eligible
)
def _quantize_mxfp8_impl(value: Tensor, scale_exponent: Tensor) -> QuantizedTensor:
    return quantize_static(value, scale_exponent.to(value.device, torch.int8))


ops.quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_quantize_eligible
)(quantize_int8)
ops.layernorm_quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_layernorm_eligible
)(layernorm_quantize_int8)
ops.gelu_quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=lambda value, *a, **k: _row_major(value)
)(gelu_quantize_int8)
