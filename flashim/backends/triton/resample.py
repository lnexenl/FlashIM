"""Bilinear upsampling (``align_corners=False``) fused with a concatenation or an addition (``ops.upsample_concat`` /
``ops.upsample_add``): the upsampled FP32 map is never written; the output is channels-last. PIL's bicubic resize
(``ops.pil_bicubic_resize``): both separable passes and roundings, and up to two output sizes, in one launch."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
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
def _bilinear_source(dst, n_in, scale, ALIGN_CORNERS: tl.constexpr = False):
    """PyTorch's bilinear source index (``align_corners=False`` or ``True``), its two taps, and linear weights."""
    if ALIGN_CORNERS:
        src = dst.to(tl.float32) * scale
    else:
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


@triton.jit
def _upsample_add_nhwc_kernel(
    coarse_ptr, other_ptr, out_ptr,
    H, W, h, w, scale_h, scale_w,
    cs_b, cs_c, cs_h, cs_w, os_b, os_c, os_h, os_w,
    C: tl.constexpr, BLOCK_X: tl.constexpr, PC: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr, ROUND_BF16: tl.constexpr,
):
    """Row-wise bilinear upsample + add for channels-last output (supports ``align_corners=True`` and BF16 output)."""
    row = tl.program_id(1)
    b = (row // H).to(tl.int64)
    y = row % H
    x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    x_ok = x < W
    x = tl.where(x_ok, x, 0)
    y0, y1, ly0, ly1 = _bilinear_source(y, h, scale_h, ALIGN_CORNERS)
    x0, x1, lx0, lx1 = _bilinear_source(x, w, scale_w, ALIGN_CORNERS)
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
    sample = ly0 * (lx0 * v00 + lx1 * v01) + ly1 * (lx0 * v10 + lx1 * v11)
    if ROUND_BF16:
        value = sample.to(tl.bfloat16) + d.to(tl.bfloat16)
    else:
        value = (sample + d).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + ((row.to(tl.int64) * W + x) * C)[:, None] + c[None, :], value, mask=m)


def upsample_add(
    coarse: Tensor, delta: Tensor, *, align_corners: bool = False, out_dtype: torch.dtype | None = None
) -> Tensor:
    if not align_corners and out_dtype is None:
        out = torch.empty(delta.shape, device=delta.device, dtype=torch.float32, memory_format=torch.channels_last)
        return _launch(coarse, delta, out, concat=False)
    batch, channels, height, width = delta.shape
    h, w = coarse.shape[2:]
    dtype = out_dtype or torch.float32
    out = torch.empty(delta.shape, device=delta.device, dtype=dtype, memory_format=torch.channels_last)
    scale_h = (h - 1) / (height - 1) if (align_corners and height > 1) else (0.0 if align_corners else h / height)
    scale_w = (w - 1) / (width - 1) if (align_corners and width > 1) else (0.0 if align_corners else w / width)
    block_x = 8 if channels >= 256 else 16
    _upsample_add_nhwc_kernel[(triton.cdiv(width, block_x), batch * height)](
        coarse, delta, out, height, width, h, w, scale_h, scale_w, *coarse.stride(), *delta.stride(),
        C=channels, BLOCK_X=block_x, PC=triton.next_power_of_2(channels),
        ALIGN_CORNERS=align_corners, ROUND_BF16=dtype == torch.bfloat16, num_warps=4,
    )
    return out


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


def _eligible_add(
    coarse: Tensor, delta: Tensor, *, align_corners: bool = False, out_dtype: torch.dtype | None = None
) -> bool:
    if not (coarse.ndim == 4 and delta.ndim == 4 and coarse.shape[:2] == delta.shape[:2] and delta.is_cuda):
        return False
    if align_corners or out_dtype is not None:
        return delta.shape[1] <= 512 and (out_dtype in (None, torch.float32, torch.bfloat16, torch.float16))
    return True


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


def _rows_eligible_add(
    coarse: Tensor, delta: Tensor, *, align_corners: bool = False, out_dtype: torch.dtype | None = None
) -> bool:
    if align_corners or out_dtype is not None:
        return False
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


# -- PIL's bicubic image resize (ops.pil_bicubic_resize) ------------------------------------------------------------
# Bit-identical to the reference's torch ``upsample_bicubic2d_aa`` passes + 8-bit roundings on CUDA: the span / weight
# / accumulation arithmetic below is that kernel's (ATen/native/cuda/UpSample.cuh) with nvcc's FMA contractions spelled
# out (``tl.fma``) and Triton's own contraction off (``enable_fp_fusion=False``). The contraction pattern was found by
# emulating it against torch's weights (one-hot inputs, 216 size pairs: every weight bit-equal); tests/ops checks the
# outputs bit for bit. torch CUDA divides by a CPU scalar as ``x * (1 / 255)`` in FP32.
_INV255 = tl.constexpr(float(np.float32(1.0) / np.float32(255.0)))
_MAX_TAPS = 24  # width taps held in registers: downscaling by up to 5.5x (2 * ceil(2 * scale) + 1 taps)


@triton.jit
def _bicubic(x):
    """torch's ``BicubicFilterFunctor`` (a = -0.5)."""
    x = tl.abs(x)
    near = tl.fma(tl.fma(x, 1.5, -2.5) * x, x, 1.0)
    far = tl.fma(tl.fma(x - 5.0, x, 8.0), x, -4.0) * -0.5
    return tl.where(x < 1.0, near, tl.where(x < 2.0, far, 0.0))


@triton.jit
def _aa_span(i, n_in, scale, support):
    """``_compute_weights_span``: first input index, tap count, ``first - center``. nvcc contracts every use of
    ``center = scale * (i + 0.5)`` into an FMA (``center - support``, ``center + support``, ``first - center``)."""
    half = i.to(tl.float32) + 0.5
    first = tl.maximum((tl.fma(scale, half, -support) + 0.5).to(tl.int32), 0)
    size = tl.minimum((tl.fma(scale, half, support) + 0.5).to(tl.int32), n_in) - first
    return first, size, tl.fma(-scale, half, first.to(tl.float32))


@triton.jit
def _aa_total(first_m_center, size, invscale, K):
    """``_compute_weights``' sum of the unnormalized weights, in tap order."""
    total = tl.zeros(first_m_center.shape, tl.float32)
    for j in range(K):
        total += tl.where(j < size, _bicubic((j + first_m_center + 0.5) * invscale), 0.0)
    return total


@triton.jit
def _div_rn(a, b, rcp):
    """``a / b`` correctly rounded from ``rcp = RN(1 / b)``: two FMA residual corrections (after the first the
    quotient is within 1 ulp, so the second rounds correctly by Markstein's theorem; normal-range operands). Measured
    1.27x faster than ``div_rn`` for the per-row height weights (H20)."""
    q = a * rcp
    q = tl.fma(tl.fma(-q, b, a), rcp, q)
    return tl.fma(tl.fma(-q, b, a), rcp, q)


@triton.jit
def _wx(j: tl.constexpr, xn, xmc, inv_x, x_total):
    """Normalized width weight of tap ``j`` (0 past the span)."""
    w = tl.where(j < xn, _bicubic((j + xmc + 0.5) * inv_x), 0.0)
    return tl.where(x_total != 0.0, tl.math.div_rn(w, x_total), w)


@triton.jit
def _quantize8(x):
    """``(x * 255).round().clamp(0, 255) / 255`` as torch computes it on CUDA."""
    q = tl.extra.cuda.libdevice.rint(x * 255.0)
    return tl.minimum(tl.maximum(q, 0.0), 255.0) * _INV255


@triton.jit
def _pil_resize_tile(
    src, s_b, s_c, s_h, s_w, H, W, C, out, OH, OW, tile, plane,
    scale_x, support_x, inv_x, scale_y, support_y, inv_y, quant_x, quant_y,
    KX: tl.constexpr, KY, BH: tl.constexpr, BW: tl.constexpr, U8: tl.constexpr,
):
    """One ``[BH, BW]`` output tile of one plane, one output column per thread. The input rows the tile's height
    taps span are streamed in order: width pass of the row (``KX`` taps, weights computed once per program), 8-bit
    rounding, then one FMA into every output row of the tile with that row's height weight (0 outside its span: an
    exact no-op). Increasing input rows = torch's tap order, so each output is torch's FMA chain. A pass whose side
    keeps its size has identity weights (exact) and no rounding (``quant_* = 0``), as the reference skips it.

    Measured (H20, 1195 x 1600 uint8 HWC -> 760 x 1024 + 784 x 784, in-situ CUDA graph): 0.11 ms vs 0.35 ms for a
    variant holding the ``[rows, BW]`` width pass in registers and gathering the height taps from it (``tl.gather``
    put the lanes along rows: 32 sectors per load request)."""
    tiles_w = tl.cdiv(OW, BW)
    cols = (tile % tiles_w) * BW + tl.arange(0, BW)
    rows = (tile // tiles_w) * BH + tl.arange(0, BH)
    x0, xn, xmc = _aa_span(cols, W, scale_x, support_x)
    x_total = _aa_total(xmc, xn, inv_x, KX)
    ws = (_wx(0, xn, xmc, inv_x, x_total), _wx(1, xn, xmc, inv_x, x_total), _wx(2, xn, xmc, inv_x, x_total),
          _wx(3, xn, xmc, inv_x, x_total), _wx(4, xn, xmc, inv_x, x_total), _wx(5, xn, xmc, inv_x, x_total),
          _wx(6, xn, xmc, inv_x, x_total), _wx(7, xn, xmc, inv_x, x_total), _wx(8, xn, xmc, inv_x, x_total),
          _wx(9, xn, xmc, inv_x, x_total), _wx(10, xn, xmc, inv_x, x_total), _wx(11, xn, xmc, inv_x, x_total),
          _wx(12, xn, xmc, inv_x, x_total), _wx(13, xn, xmc, inv_x, x_total), _wx(14, xn, xmc, inv_x, x_total),
          _wx(15, xn, xmc, inv_x, x_total), _wx(16, xn, xmc, inv_x, x_total), _wx(17, xn, xmc, inv_x, x_total),
          _wx(18, xn, xmc, inv_x, x_total), _wx(19, xn, xmc, inv_x, x_total), _wx(20, xn, xmc, inv_x, x_total),
          _wx(21, xn, xmc, inv_x, x_total), _wx(22, xn, xmc, inv_x, x_total), _wx(23, xn, xmc, inv_x, x_total))
    y0, yn, ymc = _aa_span(rows, H, scale_y, support_y)
    y_total = _aa_total(ymc, yn, inv_y, KY)
    y_rcp = tl.math.div_rn(tl.full([BH], 1.0, tl.float32), y_total)
    row_ok = rows < OH
    y_first = tl.min(tl.where(row_ok, y0, H), axis=0)
    y_end = tl.max(tl.where(row_ok, y0 + yn, 0), axis=0)
    base = src + (plane // C).to(tl.int64) * s_b + (plane % C).to(tl.int64) * s_c
    res = tl.zeros([BH, BW], tl.float32)
    for y in range(y_first, y_end):
        row_ptr = base + y * s_h
        h = tl.zeros([BW], tl.float32)
        for j in tl.static_range(KX):
            v = tl.load(row_ptr + (x0 + j) * s_w, mask=j < xn, other=0)
            if U8:
                v = v.to(tl.float32) * _INV255
            h = tl.fma(v, ws[j], h)
        if quant_x != 0:
            h = _quantize8(h)
        jy = y - y0  # the tap index of input row y in each output row's span
        wy = tl.where((jy >= 0) & (jy < yn), _bicubic((jy + ymc + 0.5) * inv_y), 0.0)
        wy = tl.where(y_total != 0.0, _div_rn(wy, y_total, y_rcp), wy)
        res = tl.fma(h[None, :], wy[:, None], res)
    if quant_y != 0:
        res = _quantize8(res)
    out_off = plane.to(tl.int64) * OH * OW + rows[:, None] * OW + cols[None, :]
    tl.store(out + out_off, res, mask=row_ok[:, None] & (cols < OW)[None, :])


_SIZE_ARGS = ["s_b", "s_c", "s_h", "H", "W", "C", "OH_A", "OW_A", "tiles_a", "qxa", "qya", "OH_B", "OW_B", "qxb", "qyb",
              "KYA", "KYB"]


# Image sizes vary per call: no specialization on them (one compile per tap-count pair, not per image size).
@triton.jit(do_not_specialize=_SIZE_ARGS)
def _pil_resize_kernel(
    src, s_b, s_c, s_h, s_w, H, W, C,
    out_a, OH_A, OW_A, tiles_a, sxa, pxa, ixa, sya, pya, iya, qxa, qya,
    out_b, OH_B, OW_B, sxb, pxb, ixb, syb, pyb, iyb, qxb, qyb,
    KXA: tl.constexpr, KYA, KXB: tl.constexpr, KYB, BH: tl.constexpr, BW: tl.constexpr, U8: tl.constexpr,
):
    """Both output sizes of one source in one launch: programs below ``tiles_a`` write size A, the rest size B."""
    tile, plane = tl.program_id(0), tl.program_id(1)
    if tile < tiles_a:
        _pil_resize_tile(src, s_b, s_c, s_h, s_w, H, W, C, out_a, OH_A, OW_A, tile, plane,
                         sxa, pxa, ixa, sya, pya, iya, qxa, qya, KXA, KYA, BH, BW, U8)
    else:
        _pil_resize_tile(src, s_b, s_c, s_h, s_w, H, W, C, out_b, OH_B, OW_B, tile - tiles_a, plane,
                         sxb, pxb, ixb, syb, pyb, iyb, qxb, qyb, KXB, KYB, BH, BW, U8)


def _aa_params(n_in: int, n_out: int) -> tuple[float, float, float, int]:
    """Host side of torch's kernel: ``scale = (float) in / out``, support ``2 * scale`` (2 when upsampling),
    ``invscale`` (FP64 ``1 / scale`` rounded to FP32), the tap count ``2 * ceil(support) + 1``."""
    scale = np.float32(n_in) / np.float32(n_out)
    support = np.float32(2.0 * float(scale)) if scale >= 1 else np.float32(2.0)
    inv = np.float32(1.0 / float(scale)) if scale >= 1 else np.float32(1.0)
    return float(scale), float(support), float(inv), math.ceil(support) * 2 + 1


# 16 x 64 tiles (2 warps): measured best of 8 / 16 / 32 rows x 32 / 64 / 128 columns on H20 (0.111 ms, 1600 px).
_BH, _WARPS = 16, 2
_BW = 32 * _WARPS


def pil_bicubic_resize(image: Tensor, sizes: Sequence[tuple[int, int]]) -> tuple[Tensor, ...]:
    batch, channels, height, width = image.shape
    out = []
    for start in range(0, len(sizes), 2):
        pair = list(sizes[start:start + 2])
        outs = [torch.empty(batch, channels, h, w, device=image.device, dtype=torch.float32) for h, w in pair]
        params, tiles, taps = [], [], []
        for (h, w), o in zip(pair, outs):
            sx, px, ix, kx = _aa_params(width, w)
            sy, py, iy, ky = _aa_params(height, h)
            params.append((o, h, w, sx, px, ix, sy, py, iy, int(w != width), int(h != height)))
            tiles.append(triton.cdiv(h, _BH) * triton.cdiv(w, _BW))
            taps += [kx, ky]
        if len(pair) == 1:
            params.append(params[0])
            taps += taps
        (oa, *pa), (ob, *pb) = params
        _pil_resize_kernel[(sum(tiles[:len(pair)]), batch * channels)](
            image, *image.stride(), height, width, channels, oa, *pa[:2], tiles[0], *pa[2:], ob, *pb, *taps,
            BH=_BH, BW=_BW, U8=image.dtype == torch.uint8, num_warps=_WARPS, enable_fp_fusion=False,
        )
        out += outs
    return tuple(out)


def _eligible_pil_resize(image: Tensor, sizes: Sequence[tuple[int, int]]) -> bool:
    return (
        image.is_cuda and image.ndim == 4 and image.dtype in (torch.uint8, torch.float32) and image.numel() < 2**31
        and all(h > 0 and w > 0 and _aa_params(image.shape[-1], w)[3] <= _MAX_TAPS for h, w in sizes)
    )


ops.pil_bicubic_resize.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible_pil_resize
)(pil_bicubic_resize)


# -- strided spatial window gather -----------------------------------------------------------------------


@triton.jit
def _gather_windows_kernel(
    feat_ptr,
    ids_ptr,
    out_ptr,
    sc,
    sh,
    sw,
    H,
    W,
    COARSE_W,
    STRIDE: tl.constexpr,
    WIN: tl.constexpr,
    HALF: tl.constexpr,
    C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One program per ``(match m, window row wy)``: gathers ``WIN`` horizontal pixels x ``C`` channels from
    ``feat [1, C, H, W]`` (arbitrary strides) into ``out [M, WIN*WIN, C]`` contiguous."""
    m = tl.program_id(0).to(tl.int64)
    wy = tl.program_id(1)
    cell = tl.load(ids_ptr + m).to(tl.int64)
    cy = cell // COARSE_W
    cx = cell - cy * COARSE_W
    iy = cy * STRIDE + wy - HALF
    wx = tl.arange(0, BLOCK_W)
    c = tl.arange(0, BLOCK_C)
    ix = cx * STRIDE + wx - HALF
    in_y = (iy >= 0) & (iy < H)
    ok_w = wx < WIN
    ok_c = c < C
    in_bounds = in_y & ok_w & (ix >= 0) & (ix < W)
    ptrs = feat_ptr + iy * sh + ix[:, None] * sw + c[None, :] * sc
    vals = tl.load(ptrs, mask=in_bounds[:, None] & ok_c[None, :], other=0.0)
    out_base = out_ptr + (m * (WIN * WIN) + wy * WIN) * C
    tl.store(out_base + wx[:, None] * C + c[None, :], vals, mask=ok_w[:, None] & ok_c[None, :])


def gather_windows(feat: Tensor, ids: Tensor, coarse_w: int, win: int, stride: int) -> Tensor:
    m = ids.shape[0]
    c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
    if m == 0:
        return feat.new_empty((0, win * win, c))
    out = torch.empty((m, win * win, c), device=feat.device, dtype=feat.dtype)
    block_w = triton.next_power_of_2(win)
    block_c = triton.next_power_of_2(c)
    _gather_windows_kernel[(m, win)](
        feat,
        ids,
        out,
        feat.stride(1),
        feat.stride(2),
        feat.stride(3),
        h,
        w,
        coarse_w,
        STRIDE=stride,
        WIN=win,
        HALF=win // 2,
        C=c,
        BLOCK_W=block_w,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def _gather_windows_eligible(feat: Tensor, ids: Tensor, coarse_w: int, win: int, stride: int) -> bool:
    return (
        not torch.is_grad_enabled()
        and feat.ndim == 4
        and feat.shape[0] == 1
        and ids.ndim == 1
        and feat.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and feat.shape[1] <= 512
        and 1 <= win <= 9
        and win % 2 == 1
    )


ops.gather_windows.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_gather_windows_eligible
)(gather_windows)


# -- 2x bilinear upsampling (align_corners=True) for NHWC maps (DUSt3R / DPT) ----------------------------


@triton.jit
def _upsample_bilinear2x_nhwc_kernel(
    x_ptr,
    out_ptr,
    H,
    W,
    C,
    sb,
    sh,
    sw,
    ob,
    oh,
    ow,
    BLOCK_X: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Exact 2x ``align_corners=True`` bilinear upsampling on channels-last ``[B, C, H, W]`` -> ``[B, C, 2H, 2W]``.

    One program handles ``BLOCK_X`` adjacent output pixels along one row ``(b, oy)`` and ``BLOCK_C`` contiguous
    channels so every tap load is coalesced (128-bit vector loads along ``C``) and ``(y0, y1, wy0, wy1)`` are
    scalars shared across the tile. Source coordinates are computed in FP64 to match PyTorch's
    ``area_pixel_compute_source_index`` bit-accurately.
    """
    pid_x = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_by = tl.program_id(2)
    out_h = 2 * H
    out_w = 2 * W
    b = pid_by // out_h
    oy = pid_by - b * out_h

    den_h = out_h - 1
    den_w = out_w - 1

    num_y = oy * (H - 1)
    y0 = num_y // den_h
    y1 = tl.minimum(y0 + 1, H - 1)
    wy1 = tl.div_rn((num_y - y0 * den_h).to(tl.float32), den_h.to(tl.float32))
    wy0 = 1.0 - wy1

    ox = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    ox_ok = ox < out_w
    num_x = ox * (W - 1)
    x0 = num_x // den_w
    x1 = tl.minimum(x0 + 1, W - 1)
    wx1 = tl.div_rn((num_x - x0 * den_w).to(tl.float32), den_w.to(tl.float32))
    wx0 = 1.0 - wx1

    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = c < C
    ok = ox_ok[:, None] & c_ok[None, :]

    base_y0 = x_ptr + b.to(tl.int64) * sb + y0.to(tl.int64) * sh
    base_y1 = x_ptr + b.to(tl.int64) * sb + y1.to(tl.int64) * sh
    x0_off = x0[:, None].to(tl.int64) * sw + c[None, :]
    x1_off = x1[:, None].to(tl.int64) * sw + c[None, :]

    v00 = tl.load(base_y0 + x0_off, mask=ok, other=0.0).to(tl.float32)
    v01 = tl.load(base_y0 + x1_off, mask=ok, other=0.0).to(tl.float32)
    v10 = tl.load(base_y1 + x0_off, mask=ok, other=0.0).to(tl.float32)
    v11 = tl.load(base_y1 + x1_off, mask=ok, other=0.0).to(tl.float32)

    val = wy0 * (wx0[:, None] * v00 + wx1[:, None] * v01) + wy1 * (wx0[:, None] * v10 + wx1[:, None] * v11)
    out_off = b.to(tl.int64) * ob + oy.to(tl.int64) * oh + ox[:, None].to(tl.int64) * ow + c[None, :]
    tl.store(out_ptr + out_off, val.to(out_ptr.dtype.element_ty), mask=ok)


def upsample_bilinear2x(x: Tensor) -> Tensor:
    """Exact 2x ``align_corners=True`` bilinear upsampling for NHWC maps.

    Tile picks measured on sm_120 (RTX 5070 Ti, FP16 NHWC, DUSt3R DPT shapes): ``(8, 256, 8 warps)`` hits 46.6 us on
    ``[1, 256, 96, 128]`` (4.43x vs PyTorch 206.7 us, ~670 GB/s); ``(16, 64, 4 warps)`` hits 92.0 us on
    ``[1, 128, 192, 256]`` (4.53x vs PyTorch 416.5 us, ~680 GB/s). Uses exact integer quotient/remainder +
    ``tl.div_rn`` (no FP64) for exact coordinates without sm_120's 1/64 FP64 penalty.
    """
    batch, channels, h, w = x.shape
    out_h, out_w = 2 * h, 2 * w
    out = torch.empty(batch, channels, out_h, out_w, device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    if channels >= 256:
        block_x, block_c, num_warps = 8, 256, 8
    elif channels >= 64:
        block_x, block_c, num_warps = 16, 64, 4
    else:
        block_x, block_c, num_warps = 16, max(16, triton.next_power_of_2(channels)), 4
    grid = (triton.cdiv(out_w, block_x), triton.cdiv(channels, block_c), batch * out_h)
    _upsample_bilinear2x_nhwc_kernel[grid](
        x, out, h, w, channels,
        x.stride(0), x.stride(2), x.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
        BLOCK_X=block_x, BLOCK_C=block_c, num_warps=num_warps,
    )
    return out


def _eligible_upsample_bilinear2x(x: Tensor) -> bool:
    return (
        x.is_cuda
        and x.ndim == 4
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[2] >= 2
        and x.shape[3] >= 2
        and x.stride(1) == 1
        and x.numel() * 4 < 2**31
    )


ops.upsample_bilinear2x.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible_upsample_bilinear2x
)(upsample_bilinear2x)
