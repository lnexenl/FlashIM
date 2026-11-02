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

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


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
    coarse_ptr, other_ptr, out_ptr,
    C_OTHER, C_COARSE, H, W, h, w, scale_h, scale_w,
    cs_b, cs_c, cs_h, cs_w, os_b, os_c, os_h, os_w,
    CONCAT: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """One ``[BLOCK_P pixels, BLOCK_C channels]`` tile of the channels-last output. ``CONCAT``: channels below
    ``C_OTHER`` are copied from ``other``, the rest are bilinear samples of ``coarse``; else ``out = sample + other``."""
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
    tl.store(out_ptr + out_off, value.to(out_ptr.dtype.element_ty), mask=p_ok[:, None] & (c < c_out)[None, :])


def _launch(coarse: Tensor, other: Tensor, out: Tensor, concat: bool) -> Tensor:
    batch, c_other, height, width = other.shape
    c_coarse, h, w = coarse.shape[1:]
    channels = out.shape[1]
    block_p, block_c = 64, min(64, triton.next_power_of_2(channels))
    grid = (triton.cdiv(height * width, block_p), triton.cdiv(channels, block_c), batch)
    _upsample_combine_kernel[grid](
        coarse, other, out, c_other if concat else 0, c_coarse, height, width, h, w, h / height, w / width,
        *coarse.stride(), *other.stride(), CONCAT=concat, BLOCK_P=block_p, BLOCK_C=block_c, num_warps=4,
    )
    return out


def upsample_concat(features: Tensor, coarse: Tensor) -> Tensor:
    batch, c_features, height, width = features.shape
    out = torch.empty(
        batch, c_features + coarse.shape[1], height, width, device=features.device, dtype=features.dtype,
        memory_format=torch.channels_last,
    )
    return _launch(coarse, features, out, concat=True)


def upsample_add(coarse: Tensor, delta: Tensor) -> Tensor:
    out = torch.empty(delta.shape, device=delta.device, dtype=torch.float32, memory_format=torch.channels_last)
    return _launch(coarse, delta, out, concat=False)


def _eligible_concat(features: Tensor, coarse: Tensor) -> bool:
    return features.ndim == 4 and coarse.ndim == 4 and features.shape[0] == coarse.shape[0] and features.is_cuda


def _eligible_add(coarse: Tensor, delta: Tensor) -> bool:
    return coarse.ndim == 4 and delta.ndim == 4 and coarse.shape[:2] == delta.shape[:2] and delta.is_cuda


ops.upsample_concat.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible_concat
)(upsample_concat)
ops.upsample_add.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible_add)(upsample_add)


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
