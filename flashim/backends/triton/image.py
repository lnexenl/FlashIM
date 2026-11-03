"""``jpeg_fancy_upsample`` in two passes: one program per ``BLOCK`` chroma blocks recovers ``(Cb, Cr)`` (int32 planes),
then one per ``BLOCK`` pixels recomputes luma and applies libjpeg's triangle filter and colour conversion. Integer
arithmetic only, so it matches the reference bit for bit; the 59 eager torch kernels of the reference are what it
replaces."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops import image as _ref
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature

# the reference's constants, as constexprs the kernels can read
CR_R, CB_B, CR_G, CB_G = (tl.constexpr(v) for v in (_ref.CR_R, _ref.CB_B, _ref.CR_G, _ref.CB_G))
Y_R, Y_G, Y_B = (tl.constexpr(v) for v in (_ref.Y_R, _ref.Y_G, _ref.Y_B))
CB_R, CB_GF, CR_GF, CR_BF = (tl.constexpr(v) for v in (_ref.CB_R, _ref.CB_GF, _ref.CR_GF, _ref.CR_BF))
D1_MIN, D2_MIN, D2_SPAN = (tl.constexpr(v) for v in (_ref.D1_MIN, _ref.D2_MIN, _ref.D2_SPAN))


@triton.jit
def _chroma_kernel(rgb_ptr, inv_ptr, cb_ptr, cr_ptr, h, w, wb, n_blocks, V: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = idx < n_blocks
    bi, bj = idx // wb, idx % wb
    plane = h * w
    best = tl.full((BLOCK,), -1, tl.int32)
    rs = tl.zeros((BLOCK,), tl.int32)
    gs = tl.zeros((BLOCK,), tl.int32)
    bs = tl.zeros((BLOCK,), tl.int32)
    for dv in tl.static_range(V):
        for dh in tl.static_range(2):
            y = tl.minimum(bi * V + dv, h - 1)
            x = tl.minimum(bj * 2 + dh, w - 1)
            off = y * w + x
            r = tl.load(rgb_ptr + off, mask=ok, other=0).to(tl.int32)
            g = tl.load(rgb_ptr + plane + off, mask=ok, other=0).to(tl.int32)
            b = tl.load(rgb_ptr + 2 * plane + off, mask=ok, other=0).to(tl.int32)
            clear = (r > 0) & (r < 255) & (g > 0) & (g < 255) & (b > 0) & (b < 255)
            key = (r - g - D1_MIN) * D2_SPAN + (b - g - D2_MIN)
            packed = tl.load(inv_ptr + key, mask=ok & clear, other=-1)
            best = tl.maximum(best, packed)
            rs += r
            gs += g
            bs += b
    offset = ((128 << 16) + (1 << 15) - 1) << V
    cb_est = tl.minimum(tl.maximum((-CB_R * rs - CB_GF * gs + (1 << 15) * bs + offset) >> (16 + V), 0), 255)
    cr_est = tl.minimum(tl.maximum(((1 << 15) * rs - CR_GF * gs - CR_BF * bs + offset) >> (16 + V), 0), 255)
    exact = best >= 0
    tl.store(cb_ptr + idx, tl.where(exact, best >> 8, cb_est), mask=ok)
    tl.store(cr_ptr + idx, tl.where(exact, best & 0xFF, cr_est), mask=ok)


@triton.jit
def _cr_r(c):
    return (CR_R * (c - 128) + (1 << 15)) >> 16


@triton.jit
def _cb_b(c):
    return (CB_B * (c - 128) + (1 << 15)) >> 16


@triton.jit
def _g_term(cb, cr):
    return (-CB_G * (cb - 128) + (1 << 15) - CR_G * (cr - 128)) >> 16


@triton.jit
def _fancy(c_ptr, bi, bj, ni, nj, wb, px, mask, V: tl.constexpr):
    near = tl.load(c_ptr + bi * wb + bj, mask=mask, other=0)
    side = tl.load(c_ptr + bi * wb + nj, mask=mask, other=0)
    if V == 2:
        near = 3 * near + tl.load(c_ptr + ni * wb + bj, mask=mask, other=0)
        side = 3 * side + tl.load(c_ptr + ni * wb + nj, mask=mask, other=0)
        return (3 * near + side + 8 - px) >> 4
    else:
        return (3 * near + side + 1 + px) >> 2


@triton.jit
def _rgb_kernel(rgb_ptr, cb_ptr, cr_ptr, out_ptr, h, w, hb, wb, V: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    plane = h * w
    ok = idx < plane
    yy, xx = idx // w, idx % w
    bi, bj = yy // V, xx // 2
    px = xx % 2
    nj = tl.minimum(tl.maximum(bj + 2 * px - 1, 0), wb - 1)
    ni = tl.minimum(tl.maximum(bi + 2 * (yy % 2) - 1, 0), hb - 1)
    r = tl.load(rgb_ptr + idx, mask=ok, other=0).to(tl.int32)
    g = tl.load(rgb_ptr + plane + idx, mask=ok, other=0).to(tl.int32)
    b = tl.load(rgb_ptr + 2 * plane + idx, mask=ok, other=0).to(tl.int32)
    cb = tl.load(cb_ptr + bi * wb + bj, mask=ok, other=128)
    cr = tl.load(cr_ptr + bi * wb + bj, mask=ok, other=128)
    y_est = (Y_R * r + Y_G * g + Y_B * b + (1 << 15)) >> 16
    y = tl.where((b > 0) & (b < 255), b - _cb_b(cb), y_est)
    y = tl.where((r > 0) & (r < 255), r - _cr_r(cr), y)
    y = tl.where((g > 0) & (g < 255), g - _g_term(cb, cr), y)
    cb = _fancy(cb_ptr, bi, bj, ni, nj, wb, px, ok, V)
    cr = _fancy(cr_ptr, bi, bj, ni, nj, wb, px, ok, V)
    tl.store(out_ptr + idx, tl.minimum(tl.maximum(y + _cr_r(cr), 0), 255).to(tl.uint8), mask=ok)
    tl.store(out_ptr + plane + idx, tl.minimum(tl.maximum(y + _g_term(cb, cr), 0), 255).to(tl.uint8), mask=ok)
    tl.store(out_ptr + 2 * plane + idx, tl.minimum(tl.maximum(y + _cb_b(cb), 0), 255).to(tl.uint8), mask=ok)


def jpeg_fancy_upsample(rgb: Tensor, v_factor: int, *, block: int = 1024, num_warps: int = 4) -> Tensor:
    rgb = rgb.contiguous()
    _, h, w = rgb.shape
    hb, wb = -(-h // v_factor), -(-w // 2)
    cb = torch.empty((hb, wb), device=rgb.device, dtype=torch.int32)
    cr = torch.empty_like(cb)
    out = torch.empty_like(rgb)
    _chroma_kernel[(triton.cdiv(hb * wb, block),)](
        rgb, _ref.inverse_table(rgb.device), cb, cr, h, w, wb, hb * wb, V=v_factor, BLOCK=block, num_warps=num_warps
    )
    _rgb_kernel[(triton.cdiv(h * w, block),)](rgb, cb, cr, out, h, w, hb, wb, V=v_factor, BLOCK=block,
                                              num_warps=num_warps)
    return out


# -- registrations ---------------------------------------------------------------------------------------


def _eligible(rgb: Tensor, v_factor: int) -> bool:
    return rgb.dtype == torch.uint8 and rgb.ndim == 3 and rgb.shape[0] == 3 and v_factor in (1, 2) and rgb.numel() > 0


@ops.jpeg_fancy_upsample.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _jpeg_fancy_upsample_impl(rgb: Tensor, v_factor: int) -> Tensor:
    return jpeg_fancy_upsample(rgb, v_factor)
