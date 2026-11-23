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
from flashim.backends.triton._layernorm_stats import EXACT_MULTIPLE as LN_EXACT_MULTIPLE
from flashim.backends.triton._layernorm_stats import THREADS as LN_THREADS
from flashim.backends.triton._layernorm_stats import torch_layernorm_apply, torch_layernorm_stats
from flashim.backends.triton._libdevice import gelu_erf, gelu_launch_kwargs, torch_gelu_libdevice
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature, platform_for
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
def _quantize_int8_kernel(
    value_ptr, scale_ptr, channel_scale_ptr, out_ptr, elements, channels, BLOCK: tl.constexpr,
    CHANNEL_SCALE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    value = tl.load(value_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    if CHANNEL_SCALE:
        value = value * tl.load(channel_scale_ptr + offsets % channels, mask=valid, other=0.0)
    _store_quantized(out_ptr, 0, offsets, value, scale_ptr, elements, valid, True)


def quantize_int8(value: Tensor, scale: Tensor, *, channel_scale: Tensor | None = None) -> Int8Tensor:
    out = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    if channel_scale is None and _channels_first_view(value):
        _quantize_channels_first(value, scale, out, value.shape[-1], int8=True)
        return Int8Tensor(out, scale)
    _quantize_int8_kernel[(triton.cdiv(value.numel(), 4096),)](
        value, scale, value if channel_scale is None else channel_scale, out, value.numel(), value.shape[-1],
        BLOCK=4096, CHANNEL_SCALE=channel_scale is not None, num_warps=4,
    )
    return Int8Tensor(out, scale)


@triton.jit
def _bias_relu_quantize_int8_kernel(
    raw_ptr, bias_ptr, scale_ptr, out_ptr, H, W, C: tl.constexpr, BLOCK_W: tl.constexpr
):
    """One program: all ``C`` channels of ``BLOCK_W`` pixels of one image row. ``raw`` NCHW BF16 (loaded along the
    pixels, which are contiguous), ``out`` NHWC INT8 (stored along the channels): ``+ bias`` rounded to BF16
    (``Tensor.add_``), ReLU, then ``_store_quantized``'s INT8 math."""
    row = tl.program_id(0)  # b * H + y
    b = row // H
    x = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    x_ok = x < W
    c = tl.arange(0, C)
    plane = (b * C + c.to(tl.int64))[:, None] * H * W + (row - b * H).to(tl.int64) * W
    value = tl.load(raw_ptr + plane + x[None, :], mask=x_ok[None, :], other=0.0).to(tl.float32)
    value = (value + tl.load(bias_ptr + c).to(tl.float32)[:, None]).to(tl.bfloat16)
    value = tl.maximum(value, 0.0).to(tl.float32)
    scaled = libdevice.rint(value * (1.0 / tl.load(scale_ptr)))
    quantized = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
    tl.store(out_ptr + (row * W + x[None, :]).to(tl.int64) * C + c[:, None], quantized, mask=x_ok[None, :])


def bias_relu_quantize_int8(raw: Tensor, bias: Tensor, scale: Tensor) -> Int8Tensor:
    batch, channels, height, width = raw.shape
    out = torch.empty(batch, height, width, channels, device=raw.device, dtype=torch.int8)
    # Epilogue alone on the 3 x 896 x 1200 first conv (CUDA graph, L20 / H20 ms): 32 columns 0.50 / 0.10, 64 0.40 / 0.09,
    # 128 0.36 / 0.08; 256 and 512 columns 0.34 on L20 but not faster on H20; 8 warps no better.
    block_w = 128
    _bias_relu_quantize_int8_kernel[(batch * height, triton.cdiv(width, block_w))](
        raw, bias, scale, out, height, width, C=channels, BLOCK_W=block_w, num_warps=4
    )
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

# -- INT8, dynamic per-row FP32 scale ----------------------------------------------------------------


@triton.jit
def _quantize_int8_rows_kernel(value_ptr, cs_ptr, out_ptr, scale_ptr, rows, K: tl.constexpr, BLOCK_K: tl.constexpr,
                               ROWS: tl.constexpr, CHANNEL_SCALE: tl.constexpr):
    """``ROWS`` rows per program, each read once: optional per-channel scale, amax, ``scale = amax * (1 / 127)``
    (``safe_scale``'s eps floor), then ``round(x / scale)`` with an exact (``div_rn``) division: the torch
    reference's arithmetic, bit for bit."""
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_K)
    mask = (row < rows)[:, None] & (cols < K)[None, :]
    offsets = row[:, None].to(tl.int64) * K + cols[None, :]
    value = tl.load(value_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if CHANNEL_SCALE:
        value = value * tl.load(cs_ptr + cols, mask=cols < K, other=0.0).to(tl.float32)[None, :]
    scale = tl.max(tl.abs(value), 1) * (1.0 / 127.0)
    scale = tl.where(scale > 1.1920928955078125e-07, scale, 1.1920928955078125e-07)
    scaled = libdevice.rint(tl.math.div_rn(value, scale[:, None]))
    tl.store(out_ptr + offsets, tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, scale, mask=row < rows)


def quantize_int8_rows(value: Tensor, *, channel_scale: Tensor | None = None) -> tuple[Tensor, Tensor]:
    k = value.shape[-1]
    rows = value.numel() // k
    out = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    scale = torch.empty((*value.shape[:-1], 1), device=value.device, dtype=torch.float32)
    block_k = triton.next_power_of_2(k)
    per_program = max(1, 4096 // block_k)  # >= 4096 elements per program: 1024-wide rows 4 at a time
    _quantize_int8_rows_kernel[(triton.cdiv(rows, per_program),)](
        value, channel_scale if channel_scale is not None else scale, out, scale, rows,
        K=k, BLOCK_K=block_k, ROWS=per_program, CHANNEL_SCALE=channel_scale is not None,
        num_warps=4 if block_k <= 2048 else 8,
    )
    return out, scale


@triton.jit
def _rotate_quantize_int8_rows_kernel(
    x_ptr, r_ptr, cs_ptr, out_ptr, scale_ptr,
    K: tl.constexpr, B: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, CHANNEL_SCALE: tl.constexpr,
):
    """One program per row: load ``[G, B]`` (with optional FP32 ``channel_scale``), apply the block orthogonal
    transform ``R [B, B]`` on BF16 tensor cores (FP32 accumulation; ``H_64`` entries ``+-0.125`` are exact in BF16),
    compute the row ``amax / 127`` scale, and store INT8 values + FP32 scale in one pass."""
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, G_PAD)
    b = tl.arange(0, B)
    g_ok = g < G
    cols = g[:, None] * B + b[None, :]
    x = tl.load(x_ptr + row * K + cols, mask=g_ok[:, None], other=0.0)
    if CHANNEL_SCALE:
        cs = tl.load(cs_ptr + cols, mask=g_ok[:, None], other=0.0).to(tl.float32)
        x = (x.to(tl.float32) * cs).to(tl.bfloat16)
    else:
        x = x.to(tl.bfloat16)
    r = tl.load(r_ptr + b[:, None] * B + b[None, :]).to(tl.bfloat16)
    y = tl.dot(x, r)
    y = tl.where(g_ok[:, None], y, 0.0)
    scale = tl.max(tl.abs(y)) * (1.0 / 127.0)
    scale = tl.where(scale > 1.1920928955078125e-07, scale, 1.1920928955078125e-07)
    scaled = libdevice.rint(tl.math.div_rn(y, scale))
    q = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
    tl.store(out_ptr + row * K + cols, q, mask=g_ok[:, None])
    tl.store(scale_ptr + row, scale)


def rotate_quantize_int8_rows(
    value: Tensor, matrix: Tensor, *, channel_scale: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    k = value.shape[-1]
    b = matrix.shape[0]
    g = k // b
    g_pad = max(16, triton.next_power_of_2(g))
    rows = value.numel() // k
    out = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    scale = torch.empty((*value.shape[:-1], 1), device=value.device, dtype=torch.float32)
    _rotate_quantize_int8_rows_kernel[(rows,)](
        value, matrix, channel_scale if channel_scale is not None else scale, out, scale,
        K=k, B=b, G=g, G_PAD=g_pad, CHANNEL_SCALE=channel_scale is not None,
        num_warps=4,
    )
    return out, scale


@triton.jit
def _quantize_int8_rows_heads_kernel(
    value_ptr, out_ptr, scale_ptr, total_tokens, N,
    H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr, ROWS: tl.constexpr,
):
    """``quantize_int8_rows(x.transpose(1, 2).reshape(B * N, H * D))`` directly from contiguous ``[B, H, N, D]``
    attention output without materializing the transposed BF16 copy (bit-identical)."""
    tok = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK_C)
    tok_ok = tok < total_tokens
    col_ok = col < C
    mask = tok_ok[:, None] & col_ok[None, :]
    b = (tok // N).to(tl.int64)
    n = (tok % N).to(tl.int64)
    h = col // D
    d = col % D
    src_off = ((b[:, None] * H + h[None, :]) * N + n[:, None]) * D + d[None, :]
    val = tl.load(value_ptr + src_off, mask=mask, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(val), 1) * (1.0 / 127.0)
    scale = tl.where(scale > 1.1920928955078125e-07, scale, 1.1920928955078125e-07)
    scaled = libdevice.rint(tl.math.div_rn(val, scale[:, None]))
    dst_off = tok[:, None].to(tl.int64) * C + col[None, :]
    tl.store(out_ptr + dst_off, tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8), mask=mask)
    tl.store(scale_ptr + tok, scale, mask=tok_ok)


def quantize_int8_rows_heads(value: Tensor) -> tuple[Tensor, Tensor]:
    batch, heads, tokens, dim = value.shape
    channels = heads * dim
    total_tokens = batch * tokens
    transposed = value.transpose(1, 2)
    if transposed.is_contiguous():
        return quantize_int8_rows(transposed.reshape(total_tokens, channels))
    out = torch.empty((total_tokens, channels), device=value.device, dtype=torch.int8)
    scale = torch.empty((total_tokens, 1), device=value.device, dtype=torch.float32)
    block_c = triton.next_power_of_2(channels)
    per_program = max(1, 4096 // block_c)
    _quantize_int8_rows_heads_kernel[(triton.cdiv(total_tokens, per_program),)](
        value, out, scale, total_tokens, tokens,
        H=heads, D=dim, C=channels, BLOCK_C=block_c, ROWS=per_program,
        num_warps=4 if block_c <= 2048 else 8,
    )
    return out, scale


@triton.jit
def _layernorm_quantize_int8_rows_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, scale_ptr, eps, N: tl.constexpr, BLOCK: tl.constexpr, T: tl.constexpr,
    ITERS: tl.constexpr,
):
    """One program per row: ATen's LayerNorm statistics and affine step (:mod:`._layernorm_stats`), the result rounded
    to BF16, then the per-row quantization of :func:`_quantize_int8_rows_kernel` on that rounded row."""
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    mean, var = torch_layernorm_stats(x_ptr, base, N, T, ITERS)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = torch_layernorm_apply(x, w, b, mean, var, eps).to(tl.bfloat16).to(tl.float32)
    scale = tl.max(tl.abs(tl.where(mask, y, 0.0)), 0) * (1.0 / 127.0)
    scale = tl.where(scale > 1.1920928955078125e-07, scale, 1.1920928955078125e-07)
    scaled = libdevice.rint(tl.math.div_rn(y, scale))
    tl.store(out_ptr + base + cols, tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, scale)


def layernorm_quantize_int8_rows(value: Tensor, weight: Tensor, bias: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    n = value.shape[-1]
    rows = value.numel() // n
    out = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    scale = torch.empty((*value.shape[:-1], 1), device=value.device, dtype=torch.float32)
    _layernorm_quantize_int8_rows_kernel[(rows,)](
        value, weight, bias, out, scale, float(eps), N=n, BLOCK=triton.next_power_of_2(n), T=LN_THREADS,
        ITERS=triton.cdiv(n // 4, LN_THREADS), num_warps=LN_THREADS // 32, enable_fp_fusion=False,
    )
    return out, scale


# -- static INT8 for several consumers / from other layouts (LightGlue's INT8 glue) ----------------------


@triton.jit
def _quantize_static_int8(value, scale_ptr):
    """``quantize_int8``'s rounding (``_store_quantized``): reciprocal multiply, half to even, +-127."""
    scaled = libdevice.rint(value * (1.0 / tl.load(scale_ptr)))
    return tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)


@triton.jit
def _quantize_int8_multi_kernel(
    x_ptr, s0_ptr, s1_ptr, s2_ptr, o0_ptr, o1_ptr, o2_ptr, rows, C: tl.constexpr, LD0: tl.constexpr,
    LD1: tl.constexpr, LD2: tl.constexpr, OUTS: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """One read of ``x [rows, C]``, up to three static INT8 quantizations (each its own scale and output row stride)."""
    r = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r < rows)[:, None] & (c < C)[None, :]
    x = tl.load(x_ptr + r[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(o0_ptr + r[:, None] * LD0 + c[None, :], _quantize_static_int8(x, s0_ptr), mask=mask)
    if OUTS > 1:
        tl.store(o1_ptr + r[:, None] * LD1 + c[None, :], _quantize_static_int8(x, s1_ptr), mask=mask)
    if OUTS > 2:
        tl.store(o2_ptr + r[:, None] * LD2 + c[None, :], _quantize_static_int8(x, s2_ptr), mask=mask)


def quantize_int8_multi(value: Tensor, scales: list[Tensor], outs: list[Tensor]) -> list[Tensor]:
    rows, channels = value.shape
    (s0, o0), (s1, o1), (s2, o2) = (list(zip(scales, outs)) * 3)[:3]
    _quantize_int8_multi_kernel[(triton.cdiv(rows, 32), triton.cdiv(channels, 128))](
        value, s0, s1, s2, o0, o1, o2, rows, C=channels, LD0=o0.stride(0), LD1=o1.stride(0), LD2=o2.stride(0),
        OUTS=len(outs), BLOCK_R=32, BLOCK_C=128, num_warps=4,
    )
    return outs


@triton.jit
def _quantize_int8_heads_kernel(x_ptr, s_ptr, o_ptr, TOKENS, HEADS: tl.constexpr, D: tl.constexpr,
                                BLOCK_N: tl.constexpr):
    """``quantize_int8(x.transpose(1, 2).reshape(B * N, H * D))`` of a contiguous ``x [B, H, N, D]``."""
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    bh = tl.program_id(1)
    d = tl.arange(0, D)
    mask = (n < TOKENS)[:, None]
    x = tl.load(x_ptr + (bh.to(tl.int64) * TOKENS + n[:, None]) * D + d[None, :], mask=mask, other=0.0)
    dst = ((bh // HEADS).to(tl.int64) * TOKENS + n[:, None]) * (HEADS * D) + (bh % HEADS) * D + d[None, :]
    tl.store(o_ptr + dst, _quantize_static_int8(x.to(tl.float32), s_ptr), mask=mask)


def quantize_int8_heads(value: Tensor, scale: Tensor) -> Int8Tensor:
    batch, heads, tokens, dim = value.shape
    out = torch.empty(batch * tokens, heads * dim, device=value.device, dtype=torch.int8)
    _quantize_int8_heads_kernel[(triton.cdiv(tokens, 64), batch * heads)](
        value, scale, out, tokens, HEADS=heads, D=dim, BLOCK_N=64, num_warps=4
    )
    return Int8Tensor(out, scale)


@triton.jit
def _layernorm_gelu_quantize_int8_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, cs_ptr, out_ptr, eps, N: tl.constexpr, BLOCK: tl.constexpr, T: tl.constexpr,
    ITERS: tl.constexpr, CHANNEL_SCALE: tl.constexpr,
):
    """One program per row: ATen's LayerNorm (:mod:`._layernorm_stats`) rounded to BF16, ``nn.GELU`` rounded to BF16
    (launch with the libdevice of :mod:`._libdevice`), ``* channel_scale``, then ``quantize_int8``'s rounding."""
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    mean, var = torch_layernorm_stats(x_ptr, base, N, T, ITERS)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = torch_layernorm_apply(x, w, b, mean, var, eps).to(tl.bfloat16).to(tl.float32)
    g = gelu_erf(y).to(tl.bfloat16).to(tl.float32)
    if CHANNEL_SCALE:
        g = g * tl.load(cs_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + base + cols, _quantize_static_int8(g, scale_ptr), mask=mask)


def layernorm_gelu_quantize_int8(
    value: Tensor, weight: Tensor, bias: Tensor, eps: float, scale: Tensor, *, channel_scale: Tensor | None = None
) -> Int8Tensor:
    n = value.shape[-1]
    out = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    _layernorm_gelu_quantize_int8_kernel[(value.numel() // n,)](
        value, weight, bias, scale, channel_scale if channel_scale is not None else scale, out, float(eps), N=n,
        BLOCK=triton.next_power_of_2(n), T=LN_THREADS, ITERS=triton.cdiv(n // 4, LN_THREADS),
        CHANNEL_SCALE=channel_scale is not None, num_warps=LN_THREADS // 32, enable_fp_fusion=False,
        **gelu_launch_kwargs(value.device),
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


def _quantize_int8_eligible(value: Tensor, scale: Tensor, *, channel_scale: Tensor | None = None) -> bool:
    """With ``channel_scale``: row-major input and a contiguous FP32 ``[C]`` vector on its device."""
    if channel_scale is None:
        return _quantize_eligible(value)
    return (
        _row_major(value) and channel_scale.dtype == torch.float32 and channel_scale.is_contiguous()
        and channel_scale.shape == (value.shape[-1],) and channel_scale.device == value.device
    )


ops.quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_quantize_int8_eligible
)(quantize_int8)
ops.bias_relu_quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON],
    when=lambda raw, bias, scale: (
        raw.ndim == 4 and raw.dtype == torch.bfloat16 and raw.is_contiguous() and bias.dtype == torch.bfloat16
        and bias.shape == (raw.shape[1],) and raw.shape[1] == triton.next_power_of_2(raw.shape[1])
    ),
)(bias_relu_quantize_int8)
ops.layernorm_quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_layernorm_eligible
)(layernorm_quantize_int8)
ops.gelu_quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=lambda value, *a, **k: _row_major(value)
)(gelu_quantize_int8)
def _layernorm_rows_eligible(value, weight, bias, eps) -> bool:
    """The statistics replicate ATen's BF16 row kernel (:mod:`._layernorm_stats`): BF16 rows of ``N`` a multiple of 512
    (256 on sm_120, verified 0 diffs across N=512..2048 including N=768), with BF16 affine parameters."""
    cap = platform_for(value.device).capability
    if cap not in ((8, 9), (9, 0), (12, 0)):
        return False
    n = value.shape[-1]
    mult = 256 if cap == (12, 0) else LN_EXACT_MULTIPLE
    return (
        value.dtype == weight.dtype == torch.bfloat16 and bias is not None and bias.dtype == torch.bfloat16
        and _row_major(value) and n % mult == 0 and n <= 4096
    )


# Bit-exact against ``nn.LayerNorm`` + ``quantize_int8_rows`` as measured on L20 (sm_89), H20 (sm_90), and
# RTX 5070 Ti (sm_120): other architectures run the torch reference until checked.
ops.layernorm_quantize_int8_rows.register(
    "triton", device=DeviceSpec("cuda", min_capability=(8, 9)), requires=[Feature.TRITON],
    when=_layernorm_rows_eligible,
)(layernorm_quantize_int8_rows)
ops.quantize_int8_rows.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON],
    when=lambda value, *, channel_scale=None: (
        _row_major(value) and value.shape[-1] <= 8192
        and (channel_scale is None or (
            channel_scale.ndim == 1 and channel_scale.shape[0] == value.shape[-1]
            and channel_scale.dtype == torch.float32 and channel_scale.is_contiguous()
        ))
    ),
)(quantize_int8_rows)


def _rotate_quantize_rows_eligible(value: Tensor, matrix: Tensor, *, channel_scale: Tensor | None = None) -> bool:
    if not (_row_major(value) and matrix.ndim == 2 and matrix.is_contiguous() and matrix.shape[0] == matrix.shape[1]):
        return False
    b, k = matrix.shape[0], value.shape[-1]
    if not (16 <= b <= 128 and b == triton.next_power_of_2(b) and k % b == 0 and k <= 8192):
        return False
    return channel_scale is None or (
        channel_scale.ndim == 1
        and channel_scale.shape[0] == k
        and channel_scale.dtype == torch.float32
        and channel_scale.is_contiguous()
    )


ops.rotate_quantize_int8_rows.register(
    "triton", device=DeviceSpec("cuda", min_capability=(8, 0)), requires=[Feature.TRITON],
    when=_rotate_quantize_rows_eligible,
)(rotate_quantize_int8_rows)
ops.quantize_int8_rows_heads.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON],
    when=lambda value: (
        value.ndim == 4
        and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and value.numel() > 0
        and value.shape[1] * value.shape[3] <= 8192
        and (value.is_contiguous() or value.transpose(1, 2).is_contiguous())
    ),
)(quantize_int8_rows_heads)


def _quantize_multi_eligible(value: Tensor, scales: list[Tensor], outs: list[Tensor]) -> bool:
    return (
        value.ndim == 2 and _row_major(value) and 1 <= len(outs) == len(scales) <= 3
        and all(o.dtype == torch.int8 and o.shape == value.shape and o.stride(1) == 1 for o in outs)
        and all(s.numel() == 1 and s.dtype == torch.float32 for s in scales)
    )


def _quantize_heads_eligible(value: Tensor, scale: Tensor) -> bool:
    dim = value.shape[-1] if value.ndim == 4 else 0
    return value.ndim == 4 and _row_major(value) and dim >= 16 and dim == triton.next_power_of_2(dim)


def _layernorm_gelu_eligible(value, weight, bias, eps, scale, *, channel_scale=None) -> bool:
    """``nn.LayerNorm``-exact statistics as :func:`_layernorm_rows_eligible`, a libdevice under which the fused GELU is
    torch's, and ``channel_scale`` as :func:`_quantize_int8_eligible` takes it."""
    return (
        _layernorm_rows_eligible(value, weight, bias, eps) and torch_gelu_libdevice(value.device) is not None
        and (channel_scale is None or _quantize_int8_eligible(value, scale, channel_scale=channel_scale))
    )


ops.quantize_int8_multi.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_quantize_multi_eligible
)(quantize_int8_multi)
ops.quantize_int8_heads.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_quantize_heads_eligible
)(quantize_int8_heads)
# Exact LayerNorm statistics measured on sm_89 / sm_90 only (as layernorm_quantize_int8_rows).
ops.layernorm_gelu_quantize_int8.register(
    "triton", device=DeviceSpec("cuda", min_capability=(8, 9), max_capability=(9, 0)), requires=[Feature.TRITON],
    when=_layernorm_gelu_eligible,
)(layernorm_gelu_quantize_int8)


# -- residual + LayerNorm and cat([x, LayerNorm(value)]) (+ quantize) --------------------------------


@triton.jit
def _layernorm_add_kernel(
    res_ptr, val_ptr, w_ptr, b_ptr, out_ptr, eps,
    C: tl.constexpr, BLOCK_C: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, ROUND: tl.constexpr,
):
    """Single-pass ``residual + LayerNorm(value)`` for ``C <= 1024`` (one program per row)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C
    base = row * C + cols
    v = tl.load(val_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(v, axis=0) / C
    diff = tl.where(mask, v - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / C + tl.cast(eps, tl.float32))
    y = diff * rstd
    if HAS_WEIGHT:
        y = y * tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        y = y + tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = _round_to_dense(y, ROUND)
    r = tl.load(res_ptr + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base, (r + y).to(out_ptr.dtype.element_ty), mask=mask)


def layernorm_add(
    residual: Tensor, value: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
) -> Tensor:
    c = value.shape[-1]
    out = torch.empty_like(value)
    rows = value.numel() // c
    _layernorm_add_kernel[(rows,)](
        residual, value,
        weight if weight is not None else value,
        bias if bias is not None else value,
        out, float(eps),
        C=c, BLOCK_C=triton.next_power_of_2(c),
        HAS_WEIGHT=weight is not None, HAS_BIAS=bias is not None,
        ROUND=_ROUND_CODE[value.dtype],
        num_warps=4 if c >= 256 else 1,
    )
    return out


@triton.jit
def _layernorm_cat_kernel(
    x_ptr, val_ptr, w_ptr, b_ptr, scale_ptr, cs_ptr, out_ptr, row_scale_ptr, eps,
    C: tl.constexpr, BLOCK_C: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_CHANNEL_SCALE: tl.constexpr,
    ROUND: tl.constexpr, MODE: tl.constexpr,
):
    """Single-pass ``cat([x, LayerNorm(value)], dim=-1)`` (``C <= 1024``, one program per row).

    ``MODE``: 0 = dense output, 1 = static INT8, 2 = dynamic per-row INT8, 3 = static MXFP8 (requires ``C % 32 == 0``).
    When ``HAS_CHANNEL_SCALE`` is set, multiplies ``[x, y]`` by per-channel FP32 ``cs_ptr`` (SmoothQuant ``1 / s``)
    before quantization.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C
    base = row * C + cols
    v = tl.load(val_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(v, axis=0) / C
    diff = tl.where(mask, v - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / C + tl.cast(eps, tl.float32))
    y = diff * rstd
    if HAS_WEIGHT:
        y = y * tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        y = y + tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = _round_to_dense(y, ROUND)
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    if HAS_CHANNEL_SCALE:
        x = x * tl.load(cs_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = y * tl.load(cs_ptr + C + cols, mask=mask, other=1.0).to(tl.float32)

    out_left = out_ptr + row * (2 * C) + cols
    out_right = out_ptr + row * (2 * C) + C + cols
    if MODE == 0:
        tl.store(out_left, x.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_right, y.to(out_ptr.dtype.element_ty), mask=mask)
    elif MODE == 1:
        inv = 1.0 / tl.load(scale_ptr)
        qx = tl.maximum(tl.minimum(libdevice.rint(x * inv), 127.0), -127.0).to(tl.int8)
        qy = tl.maximum(tl.minimum(libdevice.rint(y * inv), 127.0), -127.0).to(tl.int8)
        tl.store(out_left, qx, mask=mask)
        tl.store(out_right, qy, mask=mask)
    elif MODE == 2:
        ax = tl.max(tl.where(mask, tl.abs(x), 0.0), axis=0)
        ay = tl.max(tl.where(mask, tl.abs(y), 0.0), axis=0)
        scale = tl.maximum(tl.maximum(ax, ay) * (1.0 / 127.0), 1e-8)
        tl.store(row_scale_ptr + row, scale)
        inv = 1.0 / scale
        qx = tl.maximum(tl.minimum(libdevice.rint(x * inv), 127.0), -127.0).to(tl.int8)
        qy = tl.maximum(tl.minimum(libdevice.rint(y * inv), 127.0), -127.0).to(tl.int8)
        tl.store(out_left, qx, mask=mask)
        tl.store(out_right, qy, mask=mask)
    else:
        exp_l = tl.load(scale_ptr + cols // 32, mask=mask, other=0).to(tl.float32)
        exp_r = tl.load(scale_ptr + (C + cols) // 32, mask=mask, other=0).to(tl.float32)
        qx = to_e4m3(tl.maximum(tl.minimum(x * tl.exp2(-exp_l), 448.0), -448.0))
        qy = to_e4m3(tl.maximum(tl.minimum(y * tl.exp2(-exp_r), 448.0), -448.0))
        tl.store(out_left, qx, mask=mask)
        tl.store(out_right, qy, mask=mask)


def _launch_layernorm_cat(
    x: Tensor, value: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float,
    mode: int, out: Tensor, scale: Tensor | None = None, row_scale: Tensor | None = None,
    channel_scale: Tensor | None = None,
) -> None:
    c = value.shape[-1]
    rows = value.numel() // c
    _layernorm_cat_kernel[(rows,)](
        x, value,
        weight if weight is not None else value,
        bias if bias is not None else value,
        scale if scale is not None else value,
        channel_scale if channel_scale is not None else value,
        out,
        row_scale if row_scale is not None else value,
        float(eps),
        C=c, BLOCK_C=triton.next_power_of_2(c),
        HAS_WEIGHT=weight is not None, HAS_BIAS=bias is not None,
        HAS_CHANNEL_SCALE=channel_scale is not None,
        ROUND=_ROUND_CODE[value.dtype], MODE=mode,
        num_warps=4 if c >= 256 else 1,
    )


def layernorm_cat(x: Tensor, value: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:
    out = torch.empty((*x.shape[:-1], 2 * x.shape[-1]), device=x.device, dtype=x.dtype)
    _launch_layernorm_cat(x, value, weight, bias, eps, 0, out)
    return out


def layernorm_cat_quantize_int8(
    x: Tensor, value: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float, scale: Tensor
) -> Int8Tensor:
    out = torch.empty((*x.shape[:-1], 2 * x.shape[-1]), device=x.device, dtype=torch.int8)
    _launch_layernorm_cat(x, value, weight, bias, eps, 1, out, scale=scale)
    return Int8Tensor(out, scale)


def layernorm_cat_quantize_int8_rows(
    x: Tensor, value: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float,
    *, channel_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    out = torch.empty((*x.shape[:-1], 2 * x.shape[-1]), device=x.device, dtype=torch.int8)
    row_scale = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    _launch_layernorm_cat(x, value, weight, bias, eps, 2, out, row_scale=row_scale, channel_scale=channel_scale)
    return out, row_scale


def layernorm_cat_quantize_mxfp8(
    x: Tensor, value: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float, scale_exponent: Tensor
) -> QuantizedTensor:
    c2 = 2 * x.shape[-1]
    out = torch.empty((*x.shape[:-1], c2), device=x.device, dtype=torch.float8_e4m3fn)
    exp = scale_exponent.to(x.device, torch.int8)
    _launch_layernorm_cat(x, value, weight, bias, eps, 3, out, scale=exp)
    return QuantizedTensor(out, exp, (*x.shape[:-1], c2))


def _ln_pair_eligible(
    a: Tensor, b: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float, *_,
    channel_scale: Tensor | None = None,
) -> bool:
    ok = (
        _row_major(a) and _row_major(b) and a.shape == b.shape and a.dtype == b.dtype and a.shape[-1] <= 1024
        and (weight is None or weight.is_contiguous()) and (bias is None or bias.is_contiguous())
    )
    if channel_scale is not None:
        ok = (
            ok and channel_scale.dtype == torch.float32 and channel_scale.is_contiguous()
            and channel_scale.shape == (2 * a.shape[-1],) and channel_scale.device == a.device
        )
    return ok


ops.layernorm_add.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_ln_pair_eligible
)(layernorm_add)
ops.layernorm_cat.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_ln_pair_eligible
)(layernorm_cat)
ops.layernorm_cat_quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_ln_pair_eligible
)(layernorm_cat_quantize_int8)
ops.layernorm_cat_quantize_int8_rows.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_ln_pair_eligible
)(layernorm_cat_quantize_int8_rows)
ops.layernorm_cat_quantize_mxfp8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3],
    when=lambda x, v, w, b, eps, se: _ln_pair_eligible(x, v, w, b, eps) and x.shape[-1] % 32 == 0,
)(layernorm_cat_quantize_mxfp8)
