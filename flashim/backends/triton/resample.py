"""Bilinear upsampling (``align_corners=False``) fused with a concatenation or an addition (``ops.upsample_concat`` /
``ops.upsample_add``): the upsampled FP32 map is never written; the output is channels-last."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature
from flashim.quant.tensor import Int8Tensor


@triton.jit
def _bilinear_source(dst, n_in, scale):
    """PyTorch's ``align_corners=False`` source index: ``max(scale * (dst + 0.5) - 0.5, 0)``, its two taps, weights."""
    src = tl.maximum(scale * (dst.to(tl.float32) + 0.5) - 0.5, 0.0)
    i0 = src.to(tl.int32)
    i1 = tl.where(i0 < n_in - 1, i0 + 1, i0)
    lam1 = src - i0.to(tl.float32)
    return i0, i1, 1.0 - lam1, lam1


@triton.jit
def _bilinear_tile(base, cs, sh, sw, c, y0, y1, x0, x1, ly0, ly1, lx0, lx1, ok):
    """``[BLOCK_P, BLOCK_C]`` bilinear samples (FP32) of channels ``c`` at the pixels' taps."""
    v00 = tl.load(base + y0[:, None] * sh + x0[:, None] * sw + c[None, :] * cs, mask=ok, other=0.0).to(tl.float32)
    v01 = tl.load(base + y0[:, None] * sh + x1[:, None] * sw + c[None, :] * cs, mask=ok, other=0.0).to(tl.float32)
    v10 = tl.load(base + y1[:, None] * sh + x0[:, None] * sw + c[None, :] * cs, mask=ok, other=0.0).to(tl.float32)
    v11 = tl.load(base + y1[:, None] * sh + x1[:, None] * sw + c[None, :] * cs, mask=ok, other=0.0).to(tl.float32)
    return ly0[:, None] * (lx0[:, None] * v00 + lx1[:, None] * v01) + ly1[:, None] * (lx0[:, None] * v10 + lx1[:, None] * v11)


@triton.jit
def _upsample_combine_kernel(
    coarse_ptr, other_ptr, out_ptr, scale_ptr,
    C_OTHER, C_COARSE, H, W, h, w, scale_h, scale_w,
    cs_b, cs_c, cs_h, cs_w, os_b, os_c, os_h, os_w,
    CONCAT: tl.constexpr, INT8: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """One ``[BLOCK_P pixels, BLOCK_C channels]`` tile of the channels-last output. ``CONCAT``: channels below
    ``C_OTHER`` are copied from ``other``, the rest are bilinear samples of ``coarse``; else ``out = sample + other``.
    ``INT8``: the BF16-rounded value quantized with the static scale at ``scale_ptr``, the rounding of
    ``ops.quantize_int8``'s Triton kernel (reciprocal multiply, half to even, +-127): bit-identical to the BF16 output
    followed by that quantize pass."""
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    b = tl.program_id(2).to(tl.int64)
    y, x = p // W, p % W
    p_ok = p < H * W
    y0, y1, ly0, ly1 = _bilinear_source(y, h, scale_h)
    x0, x1, lx0, lx1 = _bilinear_source(x, w, scale_w)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_out = C_OTHER + C_COARSE if CONCAT else C_COARSE
    out_off = ((b * H * W + p)[:, None] * c_out + c[None, :])
    other_off = b * os_b + y[:, None].to(tl.int64) * os_h + x[:, None].to(tl.int64) * os_w
    if CONCAT:
        from_other = c < C_OTHER
        copy = tl.load(other_ptr + other_off + c[None, :] * os_c, mask=p_ok[:, None] & from_other[None, :], other=0.0)
        cc = tl.where(from_other, 0, c - C_OTHER)
        ok = p_ok[:, None] & (~from_other & (c < c_out))[None, :]
        sample = _bilinear_tile(coarse_ptr + b * cs_b, cs_c, cs_h, cs_w, cc, y0, y1, x0, x1, ly0, ly1, lx0, lx1, ok)
        value = tl.where(from_other[None, :], copy.to(tl.float32), sample)
    else:
        ok = p_ok[:, None] & (c < C_COARSE)[None, :]
        sample = _bilinear_tile(coarse_ptr + b * cs_b, cs_c, cs_h, cs_w, c, y0, y1, x0, x1, ly0, ly1, lx0, lx1, ok)
        value = sample + tl.load(other_ptr + other_off + c[None, :] * os_c, mask=ok, other=0.0).to(tl.float32)
    if INT8:
        scaled = libdevice.rint(value.to(tl.bfloat16).to(tl.float32) * (1.0 / tl.load(scale_ptr)))
        value = tl.maximum(tl.minimum(scaled, 127.0), -127.0)
    tl.store(out_ptr + out_off, value.to(out_ptr.dtype.element_ty), mask=p_ok[:, None] & (c < c_out)[None, :])


def _launch(coarse: Tensor, other: Tensor, out: Tensor, concat: bool, scale: Tensor | None = None) -> Tensor:
    batch, c_other, height, width = other.shape
    c_coarse, h, w = coarse.shape[1:]
    channels = c_other + c_coarse if concat else c_coarse
    block_p, block_c = 64, min(64, triton.next_power_of_2(channels))
    grid = (triton.cdiv(height * width, block_p), triton.cdiv(channels, block_c), batch)
    _upsample_combine_kernel[grid](
        coarse, other, out, out if scale is None else scale, c_other if concat else 0, c_coarse, height, width, h, w,
        h / height, w / width, *coarse.stride(), *other.stride(), CONCAT=concat, INT8=scale is not None,
        BLOCK_P=block_p, BLOCK_C=block_c, num_warps=4,
    )
    return out


def upsample_concat(features: Tensor, coarse: Tensor) -> Tensor:
    batch, c_features, height, width = features.shape
    out = torch.empty(
        batch, c_features + coarse.shape[1], height, width, device=features.device, dtype=features.dtype,
        memory_format=torch.channels_last,
    )
    return _launch(coarse, features, out, concat=True)


def upsample_concat_int8(features: Tensor, coarse: Tensor, scale: Tensor) -> Int8Tensor:
    batch, c_features, height, width = features.shape
    out = torch.empty(batch, height, width, c_features + coarse.shape[1], device=features.device, dtype=torch.int8)
    return Int8Tensor(_launch(coarse, features, out, concat=True, scale=scale), scale)


def upsample_add(coarse: Tensor, delta: Tensor) -> Tensor:
    out = torch.empty(delta.shape, device=delta.device, dtype=torch.float32, memory_format=torch.channels_last)
    return _launch(coarse, delta, out, concat=False)


# -- exact 2x upsampling at the memory roofline (sm_89 / sm_90) ---------------------------------------------------------
# The tile kernel above recomputes the pixel -> tap math per [64 px, 64 ch] tile and its channel index
# (``where(from_other, 0, c - C_OTHER)``) hides contiguity, so the four coarse taps load as 2 B scalars (168-188
# registers, 16-25 % occupancy). The kernels below are one program per row (pair) with constexpr channel counts and
# full-channel tiles, every load issued before the first store. They compute the tile kernel's FP32 expression tree. At
# exactly 2x with BF16 taps (bilinear weights 1/4 and 3/4) every product and partial sum is exact in FP32, so no FMA
# contraction pattern can change a result and the INT8 concat is bit-identical to the tile kernel. Other ratios (the
# descriptor's 56 -> 98 map) and FP32 taps keep the tile kernel: its per-element contraction is not reproducible from
# another tiling (47-174 of 10M INT8 values differ by 1). The FP32 add is not exact; its row kernel reproduces the tile
# kernel bit for bit only empirically (tiles of at most 32 pixels: larger ones contract differently in 2-4 % of the
# outputs), so ``tests/ops`` compares the two kernels bit for bit.


@triton.jit
def _bf16_round(v):
    """FP32 -> the nearest-even BF16 value as FP32, in integer ops: ``cvt.rn.bf16.f32`` runs on the conversion pipe (1/8
    of the FP32 rate on sm_80..sm_90). Equal to the conversion for every non-NaN input."""
    u = v.to(tl.uint32, bitcast=True)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).to(tl.float32, bitcast=True)


@triton.jit
def _to_int8(value, inv):
    """``clamp(rint(value * inv), -127, 127)`` as INT8 without a conversion instruction: clamping first gives the same
    result (the bounds are integers) and ``+ 1.5 * 2**23`` rounds half to even to an integer whose two's complement is
    the low byte of the FP32 bits. Differs from the tile kernel's INT8 branch only for NaN."""
    q = tl.minimum(tl.maximum(value * inv, -127.0), 127.0) + 12582912.0
    return q.to(tl.int32, bitcast=True).to(tl.int8)


@triton.jit
def _concat_int8_pair_kernel(
    coarse_ptr, other_ptr, out_ptr, scale_ptr,
    H, W, h, w, scale_h, scale_w,
    cs_b, cs_c, cs_h, cs_w, os_b, os_c, os_h, os_w,
    C_OTHER: tl.constexpr, C_COARSE: tl.constexpr, BLOCK_X: tl.constexpr, PO: tl.constexpr, PC: tl.constexpr,
):
    """Exact 2x, BF16 inputs. Program ``(x block, i)`` writes the output rows ``2i + 1`` and ``2i + 2`` (``i = -1 .. h -
    1``; rows outside ``[0, H)`` masked). Their vertical taps are both coarse rows ``max(i, 0)`` and ``min(i + 1, h -
    1)``, so the taps and the horizontal lerps are shared and each output needs ``ly0 * top + ly1 * bottom`` (the tile
    kernel's expression and operand order). Row 0 takes taps (0, 0) with weights (1, 0) instead of (0, 1): equal for
    finite values."""
    pid = tl.program_id(1)
    b = (pid // (h + 1)).to(tl.int64)
    i = pid % (h + 1) - 1
    ya, yb = 2 * i + 1, 2 * i + 2
    a_ok, b_ok = i >= 0, yb < H
    ya_c, yb_c = tl.minimum(ya, H - 1), tl.maximum(yb, 0) * b_ok.to(tl.int32)
    x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    x_ok = x < W
    x = tl.where(x_ok, x, 0)
    C_OUT: tl.constexpr = C_OTHER + C_COARSE
    inv = 1.0 / tl.load(scale_ptr)
    xo = (x.to(tl.int64) * os_w)[:, None]
    o = tl.arange(0, PO)
    mo = x_ok[:, None] & (o < C_OTHER)[None, :]
    oo = o[None, :] * os_c
    fa = tl.load(other_ptr + b * os_b + ya_c.to(tl.int64) * os_h + xo + oo, mask=mo & a_ok, other=0.0)
    fb = tl.load(other_ptr + b * os_b + yb_c.to(tl.int64) * os_h + xo + oo, mask=mo & b_ok, other=0.0)
    _, _, ly0a, ly1a = _bilinear_source(ya_c, h, scale_h)
    _, _, ly0b, ly1b = _bilinear_source(yb_c, h, scale_h)
    x0, x1, lx0, lx1 = _bilinear_source(x, w, scale_w)
    base = coarse_ptr + b * cs_b
    rt = base + tl.maximum(i, 0).to(tl.int64) * cs_h
    rb = base + tl.minimum(i + 1, h - 1).to(tl.int64) * cs_h
    c = tl.arange(0, PC)
    m = x_ok[:, None] & (c < C_COARSE)[None, :]
    cc = c[None, :] * cs_c
    vtl = tl.load(rt + (x0 * cs_w)[:, None] + cc, mask=m, other=0.0).to(tl.float32)
    vtr = tl.load(rt + (x1 * cs_w)[:, None] + cc, mask=m, other=0.0).to(tl.float32)
    vbl = tl.load(rb + (x0 * cs_w)[:, None] + cc, mask=m, other=0.0).to(tl.float32)
    vbr = tl.load(rb + (x1 * cs_w)[:, None] + cc, mask=m, other=0.0).to(tl.float32)
    lx0, lx1 = lx0[:, None], lx1[:, None]
    top = lx0 * vtl + lx1 * vtr
    bottom = lx0 * vbl + lx1 * vbr
    pa = out_ptr + ((b * H + ya_c) * W + x.to(tl.int64))[:, None] * C_OUT
    pb = out_ptr + ((b * H + yb_c) * W + x.to(tl.int64))[:, None] * C_OUT
    tl.store(pa + o[None, :], _to_int8(fa.to(tl.float32), inv), mask=mo & a_ok)
    tl.store(pb + o[None, :], _to_int8(fb.to(tl.float32), inv), mask=mo & b_ok)
    tl.store(pa + C_OTHER + c[None, :], _to_int8(_bf16_round(ly0a * top + ly1a * bottom), inv), mask=m & a_ok)
    tl.store(pb + C_OTHER + c[None, :], _to_int8(_bf16_round(ly0b * top + ly1b * bottom), inv), mask=m & b_ok)


@triton.jit
def _add_rows_kernel(
    coarse_ptr, other_ptr, out_ptr,
    H, W, h, w, scale_h, scale_w,
    cs_b, cs_c, cs_h, cs_w, os_b, os_c, os_h, os_w,
    C: tl.constexpr, BLOCK_X: tl.constexpr, PC: tl.constexpr,
):
    """``upsample(coarse) + other`` for ``BLOCK_X`` pixels of one row x all ``C`` channels (FP32, channels-last out): the
    row's vertical taps are scalars and the channels contiguous, all four taps and the addend loaded before the store."""
    row = tl.program_id(1)
    b = (row // H).to(tl.int64)
    y = row % H
    x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    x_ok = x < W
    x = tl.where(x_ok, x, 0)
    y0, y1, ly0, ly1 = _bilinear_source(y, h, scale_h)
    x0, x1, lx0, lx1 = _bilinear_source(x, w, scale_w)
    base = coarse_ptr + b * cs_b
    r0, r1 = base + y0.to(tl.int64) * cs_h, base + y1.to(tl.int64) * cs_h
    c = tl.arange(0, PC)
    m = x_ok[:, None] & (c < C)[None, :]
    cc = c[None, :] * cs_c
    v00 = tl.load(r0 + (x0 * cs_w)[:, None] + cc, mask=m, other=0.0).to(tl.float32)
    v01 = tl.load(r0 + (x1 * cs_w)[:, None] + cc, mask=m, other=0.0).to(tl.float32)
    v10 = tl.load(r1 + (x0 * cs_w)[:, None] + cc, mask=m, other=0.0).to(tl.float32)
    v11 = tl.load(r1 + (x1 * cs_w)[:, None] + cc, mask=m, other=0.0).to(tl.float32)
    d = tl.load(other_ptr + b * os_b + y.to(tl.int64) * os_h + (x.to(tl.int64) * os_w)[:, None] + c[None, :] * os_c,
                mask=m, other=0.0).to(tl.float32)
    lx0, lx1 = lx0[:, None], lx1[:, None]
    value = ly0 * (lx0 * v00 + lx1 * v01) + ly1 * (lx0 * v10 + lx1 * v11) + d
    tl.store(out_ptr + ((row.to(tl.int64) * W + x) * C)[:, None] + c[None, :], value, mask=m)


# (BLOCK_X, num_warps) per (compute capability, output channels): rot-timed sweeps on L20 / H20 (every 2x concat tile
# is within 2-3 % on L20, at the DRAM roofline; H20 is issue-bound and wants fewer pixels per program on the wide maps:
# the 512-channel kernel uses 255 registers). The add tiles must stay <= 32 pixels (see above).
_PAIR_TILES = {(8, 9): {96: (8, 8)}, (9, 0): {512: (8, 2), 256: (16, 2), 128: (16, 2), 96: (32, 1)}}
_PAIR_DEFAULT = (16, 4)
_ADD_TILES = {(8, 9): (8, 2), (9, 0): (4, 2)}
_ADD_DEFAULT = (8, 2)


def _is_2x(features: Tensor, coarse: Tensor) -> bool:
    return features.shape[-2] == 2 * coarse.shape[-2] and features.shape[-1] == 2 * coarse.shape[-1]


def upsample_concat_int8_pair(features: Tensor, coarse: Tensor, scale: Tensor) -> Int8Tensor:
    batch, c_features, height, width = features.shape
    c_coarse, h, w = coarse.shape[1:]
    out = torch.empty(batch, height, width, c_features + c_coarse, device=features.device, dtype=torch.int8)
    block_x, num_warps = _PAIR_TILES.get(torch.cuda.get_device_capability(features.device), {}).get(
        c_features + c_coarse, _PAIR_DEFAULT
    )
    _concat_int8_pair_kernel[(triton.cdiv(width, block_x), batch * (h + 1))](
        coarse, features, out, scale, height, width, h, w, h / height, w / width, *coarse.stride(), *features.stride(),
        C_OTHER=c_features, C_COARSE=c_coarse, BLOCK_X=block_x, PO=triton.next_power_of_2(c_features),
        PC=triton.next_power_of_2(c_coarse), num_warps=num_warps,
    )
    return Int8Tensor(out, scale)


def upsample_add_rows(coarse: Tensor, delta: Tensor) -> Tensor:
    batch, channels, height, width = delta.shape
    h, w = coarse.shape[2:]
    out = torch.empty(delta.shape, device=delta.device, dtype=torch.float32, memory_format=torch.channels_last)
    block_x, num_warps = _ADD_TILES.get(torch.cuda.get_device_capability(delta.device), _ADD_DEFAULT)
    _add_rows_kernel[(triton.cdiv(width, block_x), batch * height)](
        coarse, delta, out, height, width, h, w, h / height, w / width, *coarse.stride(), *delta.stride(),
        C=channels, BLOCK_X=block_x, PC=triton.next_power_of_2(channels), num_warps=num_warps,
    )
    return out


def _eligible_concat(features: Tensor, coarse: Tensor) -> bool:
    return features.ndim == 4 and coarse.ndim == 4 and features.shape[0] == coarse.shape[0] and features.is_cuda


def _eligible_add(coarse: Tensor, delta: Tensor) -> bool:
    return coarse.ndim == 4 and delta.ndim == 4 and coarse.shape[:2] == delta.shape[:2] and delta.is_cuda


ops.upsample_concat.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible_concat
)(upsample_concat)
ops.upsample_concat_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON],
    when=lambda features, coarse, scale: _eligible_concat(features, coarse) and features.dtype == torch.bfloat16,
)(upsample_concat_int8)
ops.upsample_add.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible_add)(upsample_add)


def _rows_eligible_concat(features: Tensor, coarse: Tensor, scale: Tensor) -> bool:
    return (
        _eligible_concat(features, coarse) and features.dtype == torch.bfloat16 and coarse.dtype == torch.bfloat16
        and _is_2x(features, coarse)
    )


def _rows_eligible_add(coarse: Tensor, delta: Tensor) -> bool:
    # Wide channels-last maps only (the decoders' 256-channel deltas; with 64 channels the row kernel differs in the last bit on sm_90). With NCHW operands the tile kernel's codegen
    # contracts the FP32 sum differently (last-bit differences), and with 1 channel (RIPE's logit map) the row kernel
    # differs from it in the last bit too and is 8x slower (a 1-wide channel tile).
    channels = delta.shape[1]
    return (
        _eligible_add(coarse, delta) and coarse.stride(1) == 1 and delta.stride(1) == 1 and channels >= 96
        and channels % 32 == 0
    )


# sm_89 / sm_90 only: the capabilities these kernels were run and timed on (other architectures keep the tile kernel).
_ROWS_DEVICE = DeviceSpec("cuda", min_capability=(8, 9), max_capability=(9, 0))
ops.upsample_concat_int8.register(
    "triton_rows", device=_ROWS_DEVICE, requires=[Feature.TRITON], when=_rows_eligible_concat
)(upsample_concat_int8_pair)
ops.upsample_add.register("triton_rows", device=_ROWS_DEVICE, requires=[Feature.TRITON], when=_rows_eligible_add)(
    upsample_add_rows
)
