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


# -- fast reciprocal nearest neighbors (MASt3R online-argmax dot product) --------------------------------


@triton.jit
def _argmax_dot_split_kernel(
    q_ptr,
    db_ptr,
    val_out_ptr,
    idx_out_ptr,
    Q,
    N,
    CHUNK_N,
    SPLIT_N,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PREC: tl.constexpr,
):
    """Online argmax of ``q [Q, BLOCK_D] @ db [N, BLOCK_D]^T`` (zero-padded to power-of-two ``BLOCK_D`` for
    aligned unmasked loads) over split ``pid_s`` of ``N`` without materializing the ``[Q, N]`` similarity matrix
    in DRAM. Uses branchless ``tl.max(..., return_indices=True)`` (faster and avoids Triton 3.6's sm_100 TMEM
    layout pass failure on a branched argmax)."""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < Q

    q = tl.load(
        q_ptr + offs_m[:, None].to(tl.int64) * BLOCK_D + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    n_start = pid_s * CHUNK_N
    n_end = tl.minimum(n_start + CHUNK_N, N)
    best_val = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_M,), dtype=tl.int32)

    full_end = n_start + ((n_end - n_start) // BLOCK_N) * BLOCK_N
    for j0 in range(n_start, full_end, BLOCK_N):
        cols = j0 + tl.arange(0, BLOCK_N)
        k = tl.load(db_ptr + cols[:, None].to(tl.int64) * BLOCK_D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k), input_precision=INPUT_PREC, out_dtype=tl.float32)
        tile_max, tile_idx = tl.max(s, axis=1, return_indices=True)
        better = tile_max > best_val
        best_val = tl.where(better, tile_max, best_val)
        best_idx = tl.where(better, j0 + tile_idx, best_idx)

    if full_end < n_end:
        cols = full_end + tl.arange(0, BLOCK_N)
        col_ok = cols < n_end
        k = tl.load(
            db_ptr + cols[:, None].to(tl.int64) * BLOCK_D + offs_d[None, :],
            mask=col_ok[:, None],
            other=0.0,
        )
        s = tl.where(
            col_ok[None, :],
            tl.dot(q, tl.trans(k), input_precision=INPUT_PREC, out_dtype=tl.float32),
            -float("inf"),
        )
        tile_max, tile_idx = tl.max(s, axis=1, return_indices=True)
        better = tile_max > best_val
        best_val = tl.where(better, tile_max, best_val)
        best_idx = tl.where(better, full_end + tile_idx, best_idx)

    out_off = offs_m.to(tl.int64) * SPLIT_N + pid_s
    tl.store(val_out_ptr + out_off, best_val, mask=mask_m)
    tl.store(idx_out_ptr + out_off, best_idx, mask=mask_m)


def _argmax_dot_triton(
    queries: Tensor,
    db: Tensor,
    *,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int | None = None,
    target_ctas: int | None = None,
    input_precision: str = "ieee",
) -> Tensor:
    q_count, block_d = queries.shape
    n_count = db.shape[0]
    if q_count == 0:
        return torch.empty(0, dtype=torch.int64, device=queries.device)
    bm = 32 if q_count <= 32 else (64 if q_count <= 64 and block_m > 64 else block_m)
    is_lowp = queries.dtype in (torch.float16, torch.bfloat16)
    stages = (4 if is_lowp else 3) if num_stages is None else num_stages
    ctas = (210 if is_lowp else 140) if target_ctas is None else target_ctas
    m_blocks = triton.cdiv(q_count, bm)
    # Saturate SMs even on later iterations where active query count drops to tens/hundreds.
    max_splits = max(1, min(128, n_count // (block_n * 2)))
    split_n = min(max_splits, max(1, triton.cdiv(ctas, m_blocks)))
    chunk_n = triton.cdiv(n_count, split_n)
    # Round chunk_n up to a multiple of block_n so every non-final split avoids tail masking.
    chunk_n = triton.cdiv(chunk_n, block_n) * block_n
    split_n = triton.cdiv(n_count, chunk_n)

    vals = torch.empty((q_count, split_n), device=queries.device, dtype=torch.float32)
    idxs = torch.empty((q_count, split_n), device=queries.device, dtype=torch.int32)
    _argmax_dot_split_kernel[(m_blocks, split_n)](
        queries,
        db,
        vals,
        idxs,
        q_count,
        n_count,
        chunk_n,
        split_n,
        BLOCK_D=block_d,
        BLOCK_M=bm,
        BLOCK_N=block_n,
        INPUT_PREC=input_precision,
        num_warps=num_warps,
        num_stages=stages,
    )
    if split_n == 1:
        return idxs[:, 0].to(torch.int64)
    best_split = vals.argmax(dim=1, keepdim=True)
    return idxs.gather(1, best_split)[:, 0].to(torch.int64)


def _argmax_dot_dedup(q_idx: Tensor, src: Tensor, dst: Tensor, *, already_unique: bool = False) -> Tensor:
    """Deduplicate active query indices before running the Triton dot-argmax (bit-identical; skips repeated
    pixels that many forward queries collapse onto, while avoiding radix-sort overhead on tiny or already-unique sets)."""
    if not already_unique and q_idx.numel() > 64:
        u_idx, inv = torch.unique(q_idx, return_inverse=True)
        return _argmax_dot_triton(src[u_idx], dst)[inv]
    return _argmax_dot_triton(src[q_idx], dst)


def fast_reciprocal_nn(
    desc1: Tensor,
    desc2: Tensor,
    subsample: int = 8,
    max_iter: int = 10,
    block_size: int = 8192,
    dist: str = "dot",
) -> tuple[Tensor, Tensor]:
    """Triton online-argmax ``fast_reciprocal_nn``: pads descriptors once to power-of-two width for aligned
    unmasked loads, deduplicates repeated active queries per iteration, and streams dot-argmax without
    materializing the ``[Q, N]`` similarity matrix.

    Measured on sm_120 (RTX 5070 Ti, MASt3R 384x512x24): FP32 ``ieee`` 23.20 -> 6.27 ms (3.70x, bit-identical);
    FP16 tensor cores 23.20 -> 2.22 ms (10.45x).
    """
    h1, w1, dim1 = desc1.shape
    _h2, w2, dim2 = desc2.shape
    if dim1 != dim2:
        raise ValueError(f"Descriptor dim mismatch: {dim1} vs {dim2}")
    device = desc1.device
    start = subsample // 2
    if start >= h1 or start >= w1:
        empty = torch.empty((0, 2), dtype=torch.int64, device=device)
        return empty, empty
    block_d = max(16, triton.next_power_of_2(dim1))
    pad = block_d - dim1
    pts1 = desc1.reshape(-1, dim1)
    pts2 = desc2.reshape(-1, dim2)
    if pad:
        pts1 = torch.nn.functional.pad(pts1, (0, pad))
        pts2 = torch.nn.functional.pad(pts2, (0, pad))
    pts1 = pts1.contiguous()
    pts2 = pts2.contiguous()

    ys = torch.arange(start, h1, subsample, device=device, dtype=torch.int64)
    xs = torch.arange(start, w1, subsample, device=device, dtype=torch.int64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    xy1 = (grid_x + w1 * grid_y).reshape(-1)
    xy2 = torch.full_like(xy1, -1)
    old_xy1 = xy1.clone()
    old_xy2 = xy2.clone()
    notyet = torch.ones(xy1.shape[0], dtype=torch.bool, device=device)

    for step in range(max_iter):
        active = notyet.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        xy2[active] = _argmax_dot_dedup(xy1[active], pts1, pts2, already_unique=(step == 0))
        notyet &= old_xy2 != xy2

        active = notyet.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        xy1[active] = _argmax_dot_dedup(xy2[active], pts2, pts1)
        notyet &= old_xy1 != xy1

        old_xy2.copy_(xy2)
        old_xy1.copy_(xy1)

    converged = ~notyet
    c1, c2 = xy1[converged], xy2[converged]
    if c1.numel() == 0:
        empty = torch.empty((0, 2), dtype=torch.int64, device=device)
        return empty, empty
    packed = torch.unique((c1 << 32) | c2, sorted=True)
    idx1 = packed >> 32
    idx2 = packed & 0xFFFFFFFF
    out1 = torch.stack((idx1 % w1, idx1 // w1), dim=-1)
    out2 = torch.stack((idx2 % w2, idx2 // w2), dim=-1)
    return out1, out2


# -- exact 3D L2 reciprocal nearest neighbors (DUSt3R Morton-AABB) ---------------------------------------


@triton.jit
def _morton3d_kernel(
    pts_ptr,
    code_ptr,
    n,
    lo_x,
    lo_y,
    lo_z,
    inv_x,
    inv_y,
    inv_z,
    BLOCK: tl.constexpr,
):
    """30-bit Z-order Morton code (10 bits/axis) on ``signed_log1p(pts)`` so exponential depth ranges
    (DUSt3R ``depth_mode=exp``) get uniform spatial resolution. Only affects sort order, never distances."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = offs.to(tl.int64) * 3
    x = tl.load(pts_ptr + base, mask=mask, other=0.0)
    y = tl.load(pts_ptr + base + 1, mask=mask, other=0.0)
    z = tl.load(pts_ptr + base + 2, mask=mask, other=0.0)

    ux = tl.where(x >= 0.0, tl.log(1.0 + x), -tl.log(1.0 - x))
    uy = tl.where(y >= 0.0, tl.log(1.0 + y), -tl.log(1.0 - y))
    uz = tl.where(z >= 0.0, tl.log(1.0 + z), -tl.log(1.0 - z))

    ix = tl.minimum(tl.maximum(((ux - lo_x) * inv_x).to(tl.int32), 0), 1023)
    iy = tl.minimum(tl.maximum(((uy - lo_y) * inv_y).to(tl.int32), 0), 1023)
    iz = tl.minimum(tl.maximum(((uz - lo_z) * inv_z).to(tl.int32), 0), 1023)

    ix = (ix | (ix << 16)) & 0x030000FF
    ix = (ix | (ix << 8)) & 0x0300F00F
    ix = (ix | (ix << 4)) & 0x030C30C3
    ix = (ix | (ix << 2)) & 0x09249249

    iy = (iy | (iy << 16)) & 0x030000FF
    iy = (iy | (iy << 8)) & 0x0300F00F
    iy = (iy | (iy << 4)) & 0x030C30C3
    iy = (iy | (iy << 2)) & 0x09249249

    iz = (iz | (iz << 16)) & 0x030000FF
    iz = (iz | (iz << 8)) & 0x0300F00F
    iz = (iz | (iz << 4)) & 0x030C30C3
    iz = (iz | (iz << 2)) & 0x09249249

    tl.store(code_ptr + offs, ix | (iy << 1) | (iz << 2), mask=mask)


@triton.jit
def _gather_leaf_aabb_kernel(
    pts_ptr,
    order_ptr,
    px_ptr,
    py_ptr,
    pz_ptr,
    orig_ptr,
    lmin_x_ptr,
    lmax_x_ptr,
    lmin_y_ptr,
    lmax_y_ptr,
    lmin_z_ptr,
    lmax_z_ptr,
    n,
    LEAF: tl.constexpr,
):
    """Gather sorted 3D points into contiguous SoA ``(px, py, pz, orig_idx)`` and compute per-leaf AABBs."""
    leaf = tl.program_id(0)
    offs = leaf * LEAF + tl.arange(0, LEAF)
    valid = offs < n
    clamped = tl.minimum(offs, n - 1)
    orig = tl.load(order_ptr + clamped).to(tl.int32)
    base = orig.to(tl.int64) * 3
    x = tl.load(pts_ptr + base)
    y = tl.load(pts_ptr + base + 1)
    z = tl.load(pts_ptr + base + 2)

    tl.store(px_ptr + offs, x, mask=valid)
    tl.store(py_ptr + offs, y, mask=valid)
    tl.store(pz_ptr + offs, z, mask=valid)
    tl.store(orig_ptr + offs, orig, mask=valid)

    tl.store(lmin_x_ptr + leaf, tl.min(x, axis=0))
    tl.store(lmax_x_ptr + leaf, tl.max(x, axis=0))
    tl.store(lmin_y_ptr + leaf, tl.min(y, axis=0))
    tl.store(lmax_y_ptr + leaf, tl.max(y, axis=0))
    tl.store(lmin_z_ptr + leaf, tl.min(z, axis=0))
    tl.store(lmax_z_ptr + leaf, tl.max(z, axis=0))


@triton.jit
def _super_aabb_kernel(
    lmin_x_ptr,
    lmax_x_ptr,
    lmin_y_ptr,
    lmax_y_ptr,
    lmin_z_ptr,
    lmax_z_ptr,
    smin_x_ptr,
    smax_x_ptr,
    smin_y_ptr,
    smax_y_ptr,
    smin_z_ptr,
    smax_z_ptr,
    n_leaves,
    SUPER: tl.constexpr,
):
    """Reduce ``SUPER`` leaf AABBs into one super-block AABB."""
    sb = tl.program_id(0)
    offs = tl.minimum(sb * SUPER + tl.arange(0, SUPER), n_leaves - 1)
    tl.store(smin_x_ptr + sb, tl.min(tl.load(lmin_x_ptr + offs), axis=0))
    tl.store(smax_x_ptr + sb, tl.max(tl.load(lmax_x_ptr + offs), axis=0))
    tl.store(smin_y_ptr + sb, tl.min(tl.load(lmin_y_ptr + offs), axis=0))
    tl.store(smax_y_ptr + sb, tl.max(tl.load(lmax_y_ptr + offs), axis=0))
    tl.store(smin_z_ptr + sb, tl.min(tl.load(lmin_z_ptr + offs), axis=0))
    tl.store(smax_z_ptr + sb, tl.max(tl.load(lmax_z_ptr + offs), axis=0))


@triton.jit
def _l2_nn_3d_morton_kernel(
    qx_ptr,
    qy_ptr,
    qz_ptr,
    seed_leaf_ptr,
    px_ptr,
    py_ptr,
    pz_ptr,
    db_orig_ptr,
    lmin_x_ptr,
    lmax_x_ptr,
    lmin_y_ptr,
    lmax_y_ptr,
    lmin_z_ptr,
    lmax_z_ptr,
    smin_x_ptr,
    smax_x_ptr,
    smin_y_ptr,
    smax_y_ptr,
    smin_z_ptr,
    smax_z_ptr,
    out_idx_ptr,
    n_q,
    n_db,
    n_leaves,
    n_super,
    BLOCK_M: tl.constexpr,
    LEAF: tl.constexpr,
    SUPER: tl.constexpr,
):
    """Exact L2 nearest neighbor in 3D via 2-level AABB pruning over Morton-sorted leaves.

    Every CTA holds ``BLOCK_M`` spatially adjacent queries, seeds their upper bound from the nearest
    Morton leaves, and skips any super-block or leaf whose AABB lower bound strictly exceeds the CTA's
    worst current squared distance (with a 1e-5 relative FP32 safety margin). Exact ties break to the
    smallest original database index (bit-identical to exact brute-force L2).
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_q
    clamped_m = tl.minimum(offs_m, n_q - 1)

    qx = tl.load(qx_ptr + clamped_m)
    qy = tl.load(qy_ptr + clamped_m)
    qz = tl.load(qz_ptr + clamped_m)

    q_min_x = tl.min(qx, axis=0)
    q_max_x = tl.max(qx, axis=0)
    q_min_y = tl.min(qy, axis=0)
    q_max_y = tl.max(qy, axis=0)
    q_min_z = tl.min(qz, axis=0)
    q_max_z = tl.max(qz, axis=0)

    best_d2 = tl.full((BLOCK_M,), float("inf"), dtype=tl.float32)
    best_idx = tl.full((BLOCK_M,), 0x7FFFFFFF, dtype=tl.int32)

    offs_n = tl.arange(0, LEAF)

    # Seed from the nearest Morton leaf and its immediate neighbor to get a tight initial upper bound.
    seed = tl.load(seed_leaf_ptr + pid_m)
    seed_lo = tl.maximum(seed - 1, 0)
    seed_hi = tl.minimum(seed + 1, n_leaves - 1)
    for leaf in range(seed_lo, seed_hi + 1):
        db_offs = leaf * LEAF + offs_n
        valid_n = db_offs < n_db
        px = tl.load(px_ptr + db_offs, mask=valid_n, other=0.0)
        py = tl.load(py_ptr + db_offs, mask=valid_n, other=0.0)
        pz = tl.load(pz_ptr + db_offs, mask=valid_n, other=0.0)
        orig = tl.load(db_orig_ptr + db_offs, mask=valid_n, other=0x7FFFFFFF)
        dx = qx[:, None] - px[None, :]
        dy = qy[:, None] - py[None, :]
        dz = qz[:, None] - pz[None, :]
        d2 = tl.where(valid_n[None, :], dx * dx + dy * dy + dz * dz, float("inf"))
        tile_d2 = tl.min(d2, axis=1)
        tile_idx = tl.min(tl.where(d2 == tile_d2[:, None], orig[None, :], 0x7FFFFFFF), axis=1)
        better = (tile_d2 < best_d2) | ((tile_d2 == best_d2) & (tile_idx < best_idx))
        best_d2 = tl.where(better, tile_d2, best_d2)
        best_idx = tl.where(better, tile_idx, best_idx)

    cta_r2 = tl.max(tl.where(mask_m, best_d2, 0.0), axis=0)

    # Exact 2-level AABB-pruned scan over all super-blocks and leaves.
    for sb in range(n_super):
        sx0 = tl.load(smin_x_ptr + sb)
        sx1 = tl.load(smax_x_ptr + sb)
        sy0 = tl.load(smin_y_ptr + sb)
        sy1 = tl.load(smax_y_ptr + sb)
        sz0 = tl.load(smin_z_ptr + sb)
        sz1 = tl.load(smax_z_ptr + sb)
        sdx = tl.maximum(0.0, tl.maximum(q_min_x - sx1, sx0 - q_max_x))
        sdy = tl.maximum(0.0, tl.maximum(q_min_y - sy1, sy0 - q_max_y))
        sdz = tl.maximum(0.0, tl.maximum(q_min_z - sz1, sz0 - q_max_z))
        if sdx * sdx + sdy * sdy + sdz * sdz <= cta_r2 * 1.00001 + 1e-12:
            leaf_start = sb * SUPER
            leaf_end = tl.minimum(leaf_start + SUPER, n_leaves)
            for leaf in range(leaf_start, leaf_end):
                lx0 = tl.load(lmin_x_ptr + leaf)
                lx1 = tl.load(lmax_x_ptr + leaf)
                ly0 = tl.load(lmin_y_ptr + leaf)
                ly1 = tl.load(lmax_y_ptr + leaf)
                lz0 = tl.load(lmin_z_ptr + leaf)
                lz1 = tl.load(lmax_z_ptr + leaf)
                ldx = tl.maximum(0.0, tl.maximum(q_min_x - lx1, lx0 - q_max_x))
                ldy = tl.maximum(0.0, tl.maximum(q_min_y - ly1, ly0 - q_max_y))
                ldz = tl.maximum(0.0, tl.maximum(q_min_z - lz1, lz0 - q_max_z))
                if ldx * ldx + ldy * ldy + ldz * ldz <= cta_r2 * 1.00001 + 1e-12:
                    db_offs = leaf * LEAF + offs_n
                    valid_n = db_offs < n_db
                    px = tl.load(px_ptr + db_offs, mask=valid_n, other=0.0)
                    py = tl.load(py_ptr + db_offs, mask=valid_n, other=0.0)
                    pz = tl.load(pz_ptr + db_offs, mask=valid_n, other=0.0)
                    orig = tl.load(db_orig_ptr + db_offs, mask=valid_n, other=0x7FFFFFFF)
                    dx = qx[:, None] - px[None, :]
                    dy = qy[:, None] - py[None, :]
                    dz = qz[:, None] - pz[None, :]
                    d2 = tl.where(valid_n[None, :], dx * dx + dy * dy + dz * dz, float("inf"))
                    tile_d2 = tl.min(d2, axis=1)
                    tile_idx = tl.min(tl.where(d2 == tile_d2[:, None], orig[None, :], 0x7FFFFFFF), axis=1)
                    better = (tile_d2 < best_d2) | ((tile_d2 == best_d2) & (tile_idx < best_idx))
                    best_d2 = tl.where(better, tile_d2, best_d2)
                    best_idx = tl.where(better, tile_idx, best_idx)
                    cta_r2 = tl.max(tl.where(mask_m, best_d2, 0.0), axis=0)

    tl.store(out_idx_ptr + offs_m, best_idx, mask=mask_m)


def _prepare_morton_index(
    pts: Tensor, lo_log: tuple[float, float, float], inv_span: tuple[float, float, float], leaf: int = 64, super_b: int = 32
):
    n = pts.shape[0]
    device = pts.device
    pts_c = pts.contiguous()
    codes = torch.empty(n, dtype=torch.int32, device=device)
    _morton3d_kernel[(triton.cdiv(n, 256),)](
        pts_c, codes, n, lo_log[0], lo_log[1], lo_log[2], inv_span[0], inv_span[1], inv_span[2], BLOCK=256, num_warps=4
    )
    order = torch.argsort(codes)
    codes_s = codes[order]

    n_leaves = triton.cdiv(n, leaf)
    n_super = triton.cdiv(n_leaves, super_b)
    px = torch.empty(n, dtype=torch.float32, device=device)
    py = torch.empty(n, dtype=torch.float32, device=device)
    pz = torch.empty(n, dtype=torch.float32, device=device)
    orig = torch.empty(n, dtype=torch.int32, device=device)
    lmin_x = torch.empty(n_leaves, dtype=torch.float32, device=device)
    lmax_x = torch.empty(n_leaves, dtype=torch.float32, device=device)
    lmin_y = torch.empty(n_leaves, dtype=torch.float32, device=device)
    lmax_y = torch.empty(n_leaves, dtype=torch.float32, device=device)
    lmin_z = torch.empty(n_leaves, dtype=torch.float32, device=device)
    lmax_z = torch.empty(n_leaves, dtype=torch.float32, device=device)
    _gather_leaf_aabb_kernel[(n_leaves,)](
        pts_c, order, px, py, pz, orig, lmin_x, lmax_x, lmin_y, lmax_y, lmin_z, lmax_z, n, LEAF=leaf, num_warps=2
    )
    smin_x = torch.empty(n_super, dtype=torch.float32, device=device)
    smax_x = torch.empty(n_super, dtype=torch.float32, device=device)
    smin_y = torch.empty(n_super, dtype=torch.float32, device=device)
    smax_y = torch.empty(n_super, dtype=torch.float32, device=device)
    smin_z = torch.empty(n_super, dtype=torch.float32, device=device)
    smax_z = torch.empty(n_super, dtype=torch.float32, device=device)
    _super_aabb_kernel[(n_super,)](
        lmin_x, lmax_x, lmin_y, lmax_y, lmin_z, lmax_z,
        smin_x, smax_x, smin_y, smax_y, smin_z, smax_z,
        n_leaves, SUPER=super_b, num_warps=1,
    )
    return (
        order, codes_s, px, py, pz, orig,
        (lmin_x, lmax_x, lmin_y, lmax_y, lmin_z, lmax_z),
        (smin_x, smax_x, smin_y, smax_y, smin_z, smax_z),
        n_leaves, n_super,
    )


def _query_morton_nn(q_idx_data, db_idx_data, *, block_m: int = 32, leaf: int = 64, super_b: int = 32) -> Tensor:
    q_order, q_codes, qx, qy, qz, *_ = q_idx_data
    _, db_codes, px, py, pz, db_orig, laabb, saabb, n_leaves, n_super = db_idx_data
    n_q = qx.shape[0]
    n_db = px.shape[0]
    n_blocks = triton.cdiv(n_q, block_m)
    mid_offs = (torch.arange(n_blocks, device=qx.device, dtype=torch.int64) * block_m + block_m // 2).clamp(max=n_q - 1)
    seed_leaf = (torch.searchsorted(db_codes, q_codes[mid_offs]) // leaf).clamp(max=n_leaves - 1).to(torch.int32)

    out_sorted = torch.empty(n_q, dtype=torch.int32, device=qx.device)
    _l2_nn_3d_morton_kernel[(n_blocks,)](
        qx, qy, qz, seed_leaf,
        px, py, pz, db_orig,
        *laabb, *saabb,
        out_sorted, n_q, n_db, n_leaves, n_super,
        BLOCK_M=block_m, LEAF=leaf, SUPER=super_b, num_warps=2,
    )
    out = torch.empty(n_q, dtype=torch.int64, device=qx.device)
    out[q_order] = out_sorted.to(torch.int64)
    return out


def reciprocal_nn(pts1: Tensor, pts2: Tensor, block_size: int = 4096) -> tuple[Tensor, Tensor]:
    """Exact 3D Euclidean reciprocal nearest neighbors via Morton-sorted 2-level AABB pruning."""
    del block_size
    device = pts1.device
    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty

    lo = torch.minimum(pts1.amin(dim=0), pts2.amin(dim=0))
    hi = torch.maximum(pts1.amax(dim=0), pts2.amax(dim=0))
    lo_log = (lo.sign() * torch.log1p(lo.abs())).tolist()
    hi_log = (hi.sign() * torch.log1p(hi.abs())).tolist()
    inv_span = tuple(1023.0 / max(h - l, 1e-6) for l, h in zip(lo_log, hi_log))

    idx1_data = _prepare_morton_index(pts1, lo_log, inv_span)
    idx2_data = _prepare_morton_index(pts2, lo_log, inv_span)
    nn1_in_2 = _query_morton_nn(idx1_data, idx2_data)
    nn2_in_1 = _query_morton_nn(idx2_data, idx1_data)

    idx2 = torch.arange(pts2.shape[0], device=device, dtype=torch.int64)
    reciprocal = nn1_in_2[nn2_in_1] == idx2
    return nn2_in_1[reciprocal], idx2[reciprocal]


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


def _fast_reciprocal_nn_eligible(
    desc1: Tensor,
    desc2: Tensor,
    subsample: int = 8,
    max_iter: int = 10,
    block_size: int = 8192,
    dist: str = "dot",
) -> bool:
    return (
        dist == "dot"
        and desc1.ndim == 3
        and desc2.ndim == 3
        and desc1.shape[2] == desc2.shape[2]
        and 1 <= desc1.shape[2] <= 128
        and desc1.dtype == desc2.dtype
        and desc1.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and desc1.shape[0] * desc1.shape[1] < 2**31
        and desc2.shape[0] * desc2.shape[1] < 2**31
    )


ops.fast_reciprocal_nn.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_fast_reciprocal_nn_eligible
)(fast_reciprocal_nn)


def _reciprocal_nn_eligible(pts1: Tensor, pts2: Tensor, block_size: int = 4096) -> bool:
    return (
        pts1.ndim == 2
        and pts2.ndim == 2
        and pts1.shape[1] == 3
        and pts2.shape[1] == 3
        and pts1.dtype == torch.float32
        and pts2.dtype == torch.float32
        and 0 < pts1.shape[0] < 2**31
        and 0 < pts2.shape[0] < 2**31
    )


ops.reciprocal_nn.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_reciprocal_nn_eligible
)(reciprocal_nn)
