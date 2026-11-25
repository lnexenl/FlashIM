"""Local correlation around a warp; mutual dual-softmax matches of a similarity matrix without the ``[L, S]``
probability intermediates."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.ops.correlation import local_window
from flashim.platforms import Feature, platform_for

# -- local correlation ---------------------------------------------------------------------------------------


@triton.jit
def _local_corr_kernel(
    f0_ptr, f1_ptr, warp_ptr, window_ptr, out_ptr, rows, scale,
    s0b, s0c, s0h, s0w, s1b, s1c, s1h, s1w, swb, swh, sww, swc,
    HEIGHT: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr, CC: tl.constexpr, R: tl.constexpr,
    TT: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Local correlation ``[B, K, H, W]``:
    ``out[b, k, y, x] = scale * f0[b, :, y, x] . bilinear(f1[b], warp[b, y, x] + window[k])``.

    The window offsets are whole feature pixels, so the taps share their integer corners: the dot products
    with the ``(2R+2)^2`` corners around the warp point are computed once and blended into the ``(2R+1)^2``
    taps (one gather per corner instead of four per tap). Corner row ``v`` feeds tap rows ``v - 1`` and ``v``,
    so only those two ``[BLOCK_M, TT]`` accumulators are live and each tap row is stored once complete: a
    ``[BLOCK_M, K]`` accumulator measured 180-1800 register spills (sm_120) and 2.2-3x the time.

    Each tap's bilinear weights come from its own normalized coordinate ``warp + window[k]`` (``window`` is
    :func:`~flashim.ops.correlation.local_window`), as ``grid_sample`` sees it, not from exact pixel offsets:
    in FP32 the two differ by ~1e-5 px, which measured 8x the reference error of the former
    ``fused-local-corr`` package at C=192. Out-of-frame corners read zero (``grid_sample`` zeros padding)."""
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row < rows
    b = row // (HEIGHT * WIDTH)
    pix = row - b * (HEIGHT * WIDTH)
    y = pix // WIDTH
    x = pix - y * WIDTH
    ch = tl.arange(0, CC)
    cmask = ch < C

    f0 = tl.load(
        f0_ptr + (b * s0b + y * s0h + x * s0w)[:, None] + ch[None, :] * s0c,
        mask=valid[:, None] & cmask[None, :], other=0.0,
    )
    wx = tl.load(warp_ptr + b * swb + y * swh + x * sww, mask=valid, other=0.0)
    wy = tl.load(warp_ptr + b * swb + y * swh + x * sww + swc, mask=valid, other=0.0)
    # grid_sample(align_corners=False): pixel = ((coord + 1) * size - 1) / 2
    x0f = tl.floor(((wx + 1.0) * WIDTH - 1.0) * 0.5)
    y0f = tl.floor(((wy + 1.0) * HEIGHT - 1.0) * 0.5)
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)

    SPAN: tl.constexpr = 2 * R + 2
    TAPS: tl.constexpr = 2 * R + 1
    t = tl.arange(0, TT)
    tmask = t < TAPS
    # Fraction of tap column t past its left corner x0 + t - R, and of tap row t past its top corner.
    shift = (t - R).to(tl.float32)[None, :]
    ox = tl.load(window_ptr + t * 2, mask=tmask, other=0.0)
    frac_x = ((wx[:, None] + ox[None, :] + 1.0) * WIDTH - 1.0) * 0.5 - (x0f[:, None] + shift)
    oy = tl.load(window_ptr + t * (2 * TAPS) + 1, mask=tmask, other=0.0)
    frac_y = ((wy[:, None] + oy[None, :] + 1.0) * HEIGHT - 1.0) * 0.5 - (y0f[:, None] + shift)
    out_base = out_ptr + (b * (TAPS * TAPS * HEIGHT * WIDTH) + pix)[:, None] + t[None, :] * (HEIGHT * WIDTH)
    store_mask = valid[:, None] & tmask[None, :]

    # Runtime loops: unrolled, the corner gathers were hoisted and spilled (C=192: 146-2292 spills on sm_120).
    lower = tl.zeros((BLOCK_M, TT), dtype=tl.float32)  # tap row v - 1 (corner row v is its bottom row)
    for v in range(SPAN):
        upper = tl.zeros((BLOCK_M, TT), dtype=tl.float32)  # tap row v (corner row v is its top row)
        wy_top = 1.0 - tl.sum(tl.where(t[None, :] == v, frac_y, 0.0), axis=1)
        wy_bottom = tl.sum(tl.where(t[None, :] == v - 1, frac_y, 0.0), axis=1)  # 0 at v = 0
        cy = y0 + (v - R)
        in_y = valid & (cy >= 0) & (cy < HEIGHT)
        for u in range(SPAN):
            cx = x0 + (u - R)
            inb = in_y & (cx >= 0) & (cx < WIDTH)
            f1 = tl.load(
                f1_ptr + (b * s1b + cy * s1h + cx * s1w)[:, None] + ch[None, :] * s1c,
                mask=inb[:, None] & cmask[None, :], other=0.0,
            )
            dot = tl.sum(f0 * f1, axis=1) * scale
            # corner column u is the left corner of tap column u and the right corner of tap column u - 1
            hx = tl.where(t[None, :] == u, 1.0 - frac_x, tl.where(t[None, :] == u - 1, frac_x, 0.0))
            hx = hx * dot[:, None]
            upper += hx * wy_top[:, None]
            lower += hx * wy_bottom[:, None]
        if v >= 1:
            tl.store(out_base + (v - 1) * TAPS * (HEIGHT * WIDTH), lower, mask=store_mask)
        lower = upper


def local_corr(
    feature0: Tensor, feature1: Tensor, warp: Tensor, radius: int, *, block_m: int = 16, num_warps: int = 4
) -> Tensor:
    """FP32, any strides (channels-last views gather their channel vectors coalesced); ``[B, K, H, W]`` out.

    16 rows x 4 warps was the fastest spill-free tile at all four RoMaV2 refiner calls (sm_120; C=192 r=3 at
    200^2 / 320^2: 0.84 / 1.87 ms, C=48 r=1 at 400^2 / 640^2: 0.19 / 0.29 ms; 3.5-5.2x the former
    ``fused-local-corr`` package)."""
    batch, channels, h, w = feature0.shape
    taps = 2 * radius + 1
    cc = triton.next_power_of_2(channels)
    out = torch.empty((batch, taps * taps, h, w), device=feature0.device, dtype=torch.float32)
    rows = batch * h * w
    window = local_window(radius, h, w, feature0.device)
    _local_corr_kernel[(triton.cdiv(rows, block_m),)](
        feature0, feature1, warp, window, out, rows, 1.0 / math.sqrt(channels),
        *feature0.stride(), *feature1.stride(), *warp.stride(),
        HEIGHT=h, WIDTH=w, C=channels, CC=cc, R=radius, TT=triton.next_power_of_2(taps),
        BLOCK_M=block_m, num_warps=num_warps,
    )
    return out


# -- dual-softmax mutual matches --------------------------------------------------------------------------


@triton.jit
def _row_lse_kernel(sim_ptr, out_ptr, L, S, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """``out[i] = logsumexp_j sim[i, j]`` (online: running maximum, rescaled sum)."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < L
    peak = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    total = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in range(0, S, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        ok = row_ok[:, None] & (cols[None, :] < S)
        tile = tl.load(sim_ptr + rows[:, None].to(tl.int64) * S + cols[None, :], mask=ok, other=-float("inf"))
        new_peak = tl.maximum(peak, tl.max(tile, axis=1))
        total = total * tl.exp(peak - new_peak) + tl.sum(tl.exp(tile - new_peak[:, None]), axis=1)
        peak = new_peak
    tl.store(out_ptr + rows, peak + tl.log(total), mask=row_ok)


@triton.jit
def _col_lse_argmax_kernel(sim_ptr, row_term_ptr, lse_ptr, index_ptr, L, S,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Per column ``j``: ``lse[j] = logsumexp_i sim[i, j]`` and ``index[j] = argmax_i 2 sim[i, j] + row_term[i]``."""
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols < S
    peak = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    total = tl.zeros((BLOCK_N,), dtype=tl.float32)
    best = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    best_index = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for start in range(0, L, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        row_ok = rows < L
        ok = row_ok[:, None] & col_ok[None, :]
        tile = tl.load(sim_ptr + rows[:, None].to(tl.int64) * S + cols[None, :], mask=ok, other=-float("inf"))
        new_peak = tl.maximum(peak, tl.max(tile, axis=0))
        total = total * tl.exp(peak - new_peak) + tl.sum(tl.exp(tile - new_peak[None, :]), axis=0)
        peak = new_peak
        score = 2.0 * tile + tl.load(row_term_ptr + rows, mask=row_ok, other=0.0)[:, None]
        value, index = tl.max(score, axis=0, return_indices=True)
        better = value > best
        best = tl.where(better, value, best)
        best_index = tl.where(better, start + index, best_index)
    tl.store(lse_ptr + cols, peak + tl.log(total), mask=col_ok)
    tl.store(index_ptr + cols, best_index, mask=col_ok)


@triton.jit
def _row_argmax_kernel(sim_ptr, col_term_ptr, row_term_ptr, value_ptr, index_ptr, L, S,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Per row ``i``: maximum over ``j`` of ``2 sim[i, j] + col_term[j]`` (plus ``row_term[i]``) and its column."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < L
    best = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    best_index = tl.zeros((BLOCK_M,), dtype=tl.int32)
    for start in range(0, S, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_ok = cols < S
        ok = row_ok[:, None] & col_ok[None, :]
        tile = tl.load(sim_ptr + rows[:, None].to(tl.int64) * S + cols[None, :], mask=ok, other=-float("inf"))
        score = 2.0 * tile + tl.load(col_term_ptr + cols, mask=col_ok, other=0.0)[None, :]
        value, index = tl.max(score, axis=1, return_indices=True)
        better = value > best
        best = tl.where(better, value, best)
        best_index = tl.where(better, start + index, best_index)
    tl.store(value_ptr + rows, best + tl.load(row_term_ptr + rows, mask=row_ok, other=0.0), mask=row_ok)
    tl.store(index_ptr + rows, best_index, mask=row_ok)


@triton.jit
def _col_lse_row_partials_kernel(sim_ptr, lse_ptr, row_peak_ptr, row_total_ptr, L, S,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """One ``BLOCK_N``-column band: ``lse[j] = logsumexp_i sim[i, j]`` (as ``_col_lse_argmax_kernel``), and for every
    row its partial log-sum-exp over the band as ``(peak, sum exp(sim - peak))`` at ``[band, i]``."""
    band = tl.program_id(0)
    cols = band * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols < S
    peak = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    total = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, L, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        row_ok = rows < L
        ok = row_ok[:, None] & col_ok[None, :]
        tile = tl.load(sim_ptr + rows[:, None].to(tl.int64) * S + cols[None, :], mask=ok, other=-float("inf"))
        new_peak = tl.maximum(peak, tl.max(tile, axis=0))
        total = total * tl.exp(peak - new_peak) + tl.sum(tl.exp(tile - new_peak[None, :]), axis=0)
        peak = new_peak
        row_peak = tl.max(tile, axis=1)
        row_total = tl.sum(tl.exp(tile - row_peak[:, None]), axis=1)
        partial = band.to(tl.int64) * L + rows
        tl.store(row_peak_ptr + partial, row_peak, mask=row_ok)
        tl.store(row_total_ptr + partial, row_total, mask=row_ok)
    tl.store(lse_ptr + cols, peak + tl.log(total), mask=col_ok)


@triton.jit
def _row_lse_reduce_kernel(row_peak_ptr, row_total_ptr, out_ptr, L, BANDS, BLOCK_M: tl.constexpr):
    """``out[i] = logsumexp`` of row ``i``'s band partials ``[BANDS, L]``."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < L
    peak = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    total = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for band in range(BANDS):
        band_peak = tl.load(row_peak_ptr + band.to(tl.int64) * L + rows, mask=row_ok, other=-float("inf"))
        band_total = tl.load(row_total_ptr + band.to(tl.int64) * L + rows, mask=row_ok, other=0.0)
        new_peak = tl.maximum(peak, band_peak)
        total = total * tl.exp(peak - new_peak) + band_total * tl.exp(band_peak - new_peak)
        peak = new_peak
    tl.store(out_ptr + rows, peak + tl.log(total), mask=row_ok)


@triton.jit
def _dual_argmax_kernel(sim_ptr, row_term_ptr, col_term_ptr, col_index_ptr, row_value_ptr, row_index_ptr, L, S,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """One ``BLOCK_N``-column band: ``col_index[j] = argmax_i 2 sim[i, j] + row_term[i]`` (as
    ``_col_lse_argmax_kernel``), and for every row the band's maximum of ``2 sim[i, j] + col_term[j]`` with its
    column, at ``[band, i]``."""
    band = tl.program_id(0)
    cols = band * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols < S
    col_term = tl.load(col_term_ptr + cols, mask=col_ok, other=0.0)
    best = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    best_index = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for start in range(0, L, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        row_ok = rows < L
        ok = row_ok[:, None] & col_ok[None, :]
        tile = tl.load(sim_ptr + rows[:, None].to(tl.int64) * S + cols[None, :], mask=ok, other=-float("inf"))
        score = 2.0 * tile + tl.load(row_term_ptr + rows, mask=row_ok, other=0.0)[:, None]
        value, index = tl.max(score, axis=0, return_indices=True)
        better = value > best
        best = tl.where(better, value, best)
        best_index = tl.where(better, start + index, best_index)
        row_value, row_index = tl.max(2.0 * tile + col_term[None, :], axis=1, return_indices=True)
        partial = band.to(tl.int64) * L + rows
        tl.store(row_value_ptr + partial, row_value, mask=row_ok)
        tl.store(row_index_ptr + partial, band * BLOCK_N + row_index, mask=row_ok)
    tl.store(col_index_ptr + cols, best_index, mask=col_ok)


@triton.jit
def _row_argmax_reduce_kernel(row_value_ptr, row_index_ptr, row_term_ptr, value_ptr, index_ptr, L, BANDS,
                              BLOCK_M: tl.constexpr):
    """Per row: the first band holding the maximum of its band partials (lowest column on ties, as a row scan)."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < L
    best = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    best_index = tl.zeros((BLOCK_M,), dtype=tl.int32)
    for band in range(BANDS):
        value = tl.load(row_value_ptr + band.to(tl.int64) * L + rows, mask=row_ok, other=-float("inf"))
        index = tl.load(row_index_ptr + band.to(tl.int64) * L + rows, mask=row_ok, other=0)
        better = value > best
        best = tl.where(better, value, best)
        best_index = tl.where(better, index, best_index)
    tl.store(value_ptr + rows, best + tl.load(row_term_ptr + rows, mask=row_ok, other=0.0), mask=row_ok)
    tl.store(index_ptr + rows, best_index, mask=row_ok)


# 25 tile configurations measured 3.97 to 5.42 ms on LoFTR's 16800 x 16500 matrix (sm_120); these give 4.03 ms.
ROW_TILE = {"block_m": 8, "block_n": 512, "num_warps": 4}
COL_TILE = {"block_m": 64, "block_n": 64, "num_warps": 4}
# Best of block_m / block_n in {32, 64, 128}, 4 / 8 warps, reduce_m 128 / 256 on the same matrix (L20): 3.78 ms.
BAND_TILE = {"block_m": 64, "block_n": 64, "num_warps": 4, "reduce_m": 128}


def dual_softmax_mutual_three_pass(
    sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Three streaming passes over ``sim`` (row LSE; column LSE + column argmax; row argmax): the score
    ``2 sim - lse_row[i] - lse_col[j] + biases`` is never materialized."""
    rows, cols = sim.shape
    device = sim.device
    row_lse = torch.empty(rows, device=device, dtype=torch.float32)
    col_lse = torch.empty(cols, device=device, dtype=torch.float32)
    best_row = torch.empty(cols, device=device, dtype=torch.int32)
    log_conf = torch.empty(rows, device=device, dtype=torch.float32)
    index = torch.empty(rows, device=device, dtype=torch.int32)
    rt, ct = ROW_TILE, COL_TILE
    _row_lse_kernel[(triton.cdiv(rows, rt["block_m"]),)](
        sim, row_lse, rows, cols, BLOCK_M=rt["block_m"], BLOCK_N=rt["block_n"], num_warps=rt["num_warps"]
    )
    row_term = -row_lse if row_bias is None else row_bias.float() - row_lse
    _col_lse_argmax_kernel[(triton.cdiv(cols, ct["block_n"]),)](
        sim, row_term, col_lse, best_row, rows, cols,
        BLOCK_M=ct["block_m"], BLOCK_N=ct["block_n"], num_warps=ct["num_warps"],
    )
    col_term = -col_lse if col_bias is None else col_bias.float() - col_lse
    _row_argmax_kernel[(triton.cdiv(rows, rt["block_m"]),)](
        sim, col_term, row_term, log_conf, index, rows, cols,
        BLOCK_M=rt["block_m"], BLOCK_N=rt["block_n"], num_warps=rt["num_warps"],
    )
    index = index.long()
    mutual = best_row[index].long() == torch.arange(rows, device=device)
    return log_conf, index, mutual


# Elements from which the two-pass variant wins on an L20: 1.07-1.34x from 256 MiB up (16800 x 16500: 4.76 -> 3.81 ms,
# 8192^2: 1.15 -> 0.86 ms), 0.81x at 187 MiB and 0.41x at 95 MiB, where the matrix stays in L2 and the three-pass row
# tiles keep more CTAs busy. The crossover depends on the L2 size, so other architectures keep three passes until measured.
TWO_PASS_MIN_ELEMENTS = 1 << 26


def dual_softmax_mutual(
    sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    if sim.numel() >= TWO_PASS_MIN_ELEMENTS and platform_for(sim.device).capability == (8, 9):
        return dual_softmax_mutual_two_pass(sim, row_bias, col_bias)
    return dual_softmax_mutual_three_pass(sim, row_bias, col_bias)


def dual_softmax_mutual_two_pass(
    sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Two streaming passes over ``sim`` in column bands (both log-sum-exps; both argmaxes) for matrices far larger
    than L2, where every pass is DRAM-bound (ncu on an L20: about 820 GB/s per pass of the three-pass version). The
    row-wise results of a band are kept as ``[bands, rows]`` partials and reduced by two small kernels. Same column
    results as three passes; the row log-sum-exp is summed in another order (LoFTR's matrix: identical matches,
    confidences within 4e-6)."""
    rows, cols = sim.shape
    device = sim.device
    tile = BAND_TILE
    bands = triton.cdiv(cols, tile["block_n"])
    launch = {"BLOCK_M": tile["block_m"], "BLOCK_N": tile["block_n"], "num_warps": tile["num_warps"]}
    reduce_grid = (triton.cdiv(rows, tile["reduce_m"]),)
    row_lse = torch.empty(rows, device=device, dtype=torch.float32)
    col_lse = torch.empty(cols, device=device, dtype=torch.float32)
    best_row = torch.empty(cols, device=device, dtype=torch.int32)
    log_conf = torch.empty(rows, device=device, dtype=torch.float32)
    index = torch.empty(rows, device=device, dtype=torch.int32)
    partial_a = torch.empty((bands, rows), device=device, dtype=torch.float32)
    partial_b = torch.empty((bands, rows), device=device, dtype=torch.float32)
    _col_lse_row_partials_kernel[(bands,)](sim, col_lse, partial_a, partial_b, rows, cols, **launch)
    _row_lse_reduce_kernel[reduce_grid](partial_a, partial_b, row_lse, rows, bands, BLOCK_M=tile["reduce_m"], num_warps=4)
    row_term = -row_lse if row_bias is None else row_bias.float() - row_lse
    col_term = -col_lse if col_bias is None else col_bias.float() - col_lse
    partial_index = partial_b.view(torch.int32)
    _dual_argmax_kernel[(bands,)](sim, row_term, col_term, best_row, partial_a, partial_index, rows, cols, **launch)
    _row_argmax_reduce_kernel[reduce_grid](
        partial_a, partial_index, row_term, log_conf, index, rows, bands, BLOCK_M=tile["reduce_m"], num_warps=4
    )
    index = index.long()
    mutual = best_row[index].long() == torch.arange(rows, device=device)
    return log_conf, index, mutual


# -- registrations ---------------------------------------------------------------------------------------

def _local_corr_eligible(feature0: Tensor, feature1: Tensor, warp: Tensor, radius: int) -> bool:
    batch, channels, h, w = feature0.shape
    return (
        feature0.dtype == feature1.dtype == warp.dtype == torch.float32
        and feature1.shape == feature0.shape
        and tuple(warp.shape) == (batch, h, w, 2)
        and channels <= 256
        and 0 <= radius <= 3
        and max(_span(feature0), _span(feature1), _span(warp), batch * (2 * radius + 1) ** 2 * h * w) < 2**31
    )


def _span(t: Tensor) -> int:
    """Elements addressed by ``t``'s strides (the kernel's offsets are int32)."""
    return 1 + sum((size - 1) * abs(stride) for size, stride in zip(t.shape, t.stride()))


ops.local_corr.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_local_corr_eligible
)(local_corr)


def _eligible(sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None) -> bool:
    return sim.ndim == 2 and sim.dtype == torch.float32 and sim.is_contiguous() and sim.numel() > 0


@ops.dual_softmax_mutual.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _dual_softmax_mutual_impl(
    sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    return dual_softmax_mutual(sim, row_bias, col_bias)


# -- fused tiled mutual-nearest-neighbour matching (ops.mutual_nn_match) ----------------------------------
# Avoids materializing the full [M, N] FP32 similarity matrix (4.2 GB at M=N=32k for RIPE): each row block
# computes [BLOCK_M, BLOCK_N] inner-product tiles in registers, tracks row max/argmax online and writes only
# [m_blocks, N] column partials (65 MB at 32k), reduced in a tiny second pass.


@triton.jit
def _mnn_tiled_kernel(
    a_ptr, b_ptr, row_val_ptr, row_idx_ptr, col_val_ptr, col_idx_ptr,
    M, N, D, sa_m, sa_k, sb_n, sb_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < M
    row_best = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_best_idx = tl.zeros((BLOCK_M,), dtype=tl.int32)
    a_row_base = a_ptr + rows[:, None].to(tl.int64) * sa_m

    for n_start in range(0, N, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        col_ok = cols < N
        b_col_base = b_ptr + cols[None, :].to(tl.int64) * sb_n
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, D, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_ok = ks < D
            a = tl.load(a_row_base + ks[None, :].to(tl.int64) * sa_k, mask=row_ok[:, None] & k_ok[None, :], other=0.0)
            b = tl.load(b_col_base + ks[:, None].to(tl.int64) * sb_k, mask=k_ok[:, None] & col_ok[None, :], other=0.0)
            if PRECISION == 2:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            elif PRECISION == 1:
                acc = tl.dot(a, b, acc, input_precision="tf32")
            else:
                acc = tl.dot(a, b, acc)
        tile = tl.where(row_ok[:, None] & col_ok[None, :], acc, -float("inf"))
        r_val, r_idx = tl.max(tile, axis=1, return_indices=True)
        better = r_val > row_best
        row_best = tl.where(better, r_val, row_best)
        row_best_idx = tl.where(better, n_start + r_idx, row_best_idx)
        c_val, c_idx = tl.max(tile, axis=0, return_indices=True)
        partial = pid_m.to(tl.int64) * N + cols
        tl.store(col_val_ptr + partial, c_val, mask=col_ok)
        tl.store(col_idx_ptr + partial, pid_m * BLOCK_M + c_idx, mask=col_ok)

    tl.store(row_val_ptr + rows, row_best, mask=row_ok)
    tl.store(row_idx_ptr + rows, row_best_idx, mask=row_ok)


@triton.jit
def _mnn_col_reduce_kernel(col_val_ptr, col_idx_ptr, nn10_ptr, N, M_BLOCKS, BLOCK_N: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols < N
    best = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for mb in range(M_BLOCKS):
        off = mb.to(tl.int64) * N + cols
        val = tl.load(col_val_ptr + off, mask=col_ok, other=-float("inf"))
        idx = tl.load(col_idx_ptr + off, mask=col_ok, other=0)
        better = val > best
        best = tl.where(better, val, best)
        best_idx = tl.where(better, idx, best_idx)
    tl.store(nn10_ptr + cols, best_idx, mask=col_ok)


# Above 8192^2 (256 MB materialized FP32), TF32 tensor cores run 4.4x faster than IEEE FP32 (9.92 vs 43.50 ms at
# 32k x 24k); at <= 8192^2 (all standard <=4096-kpt benchmarks), IEEE FP32 takes 0.56 ms (vs 0.39 ms TF32) and
# matches PyTorch's dense FP32 matmul exactly (no borderline MNN match flips).
_MNN_TF32_MIN_PAIRS = 64 * 1024 * 1024


def mutual_nn_match(desc0: Tensor, desc1: Tensor, threshold: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
    m, d = desc0.shape
    n = desc1.shape[0]
    device = desc0.device
    if m == 0 or n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, desc0.new_zeros(0, dtype=torch.float32)
    is_fp32 = desc0.dtype == torch.float32
    precision = (1 if m * n > _MNN_TF32_MIN_PAIRS else 2) if is_fp32 else 0
    block_m = 64 if (m < 2048 or is_fp32) else 128
    block_n = 64 if (n < 1024 or precision == 2) else 128
    block_k = 32 if is_fp32 else 64
    # Measured on sm_120 (32320 x 23731 x 256 BF16): (128, 128, 64, nw=4, ns=2) = 5.38 ms vs 6.57 ms at nw=8
    # (and 22.54 ms / 3.23 GB peak VRAM for the materialized FP32 reference).
    num_warps = 4
    m_blocks = triton.cdiv(m, block_m)
    row_val = torch.empty(m, device=device, dtype=torch.float32)
    row_idx = torch.empty(m, device=device, dtype=torch.int32)
    col_val = torch.empty((m_blocks, n), device=device, dtype=torch.float32)
    col_idx = torch.empty((m_blocks, n), device=device, dtype=torch.int32)
    _mnn_tiled_kernel[(m_blocks,)](
        desc0, desc1, row_val, row_idx, col_val, col_idx,
        m, n, d, desc0.stride(0), desc0.stride(1), desc1.stride(0), desc1.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, PRECISION=precision,
        num_warps=num_warps, num_stages=2,
    )
    nn10 = torch.empty(n, device=device, dtype=torch.int32)
    _mnn_col_reduce_kernel[(triton.cdiv(n, 256),)](col_val, col_idx, nn10, n, m_blocks, BLOCK_N=256, num_warps=4)
    nn01 = row_idx.long()
    ids0 = torch.arange(m, device=device)
    idx0 = torch.nonzero((nn10[nn01].long() == ids0) & (row_val >= threshold))[:, 0]
    return idx0, nn01[idx0], row_val[idx0]


def _mnn_eligible(desc0: Tensor, desc1: Tensor, threshold: float = 0.0) -> bool:
    return (
        desc0.ndim == desc1.ndim == 2
        and desc0.shape[1] == desc1.shape[1]
        and desc0.dtype == desc1.dtype
        and desc0.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and desc0.is_cuda and desc1.is_cuda
    )


ops.mutual_nn_match.register(
    "triton", device=DeviceSpec("cuda", min_capability=(8, 0)), requires=[Feature.TRITON], when=_mnn_eligible
)(mutual_nn_match)
