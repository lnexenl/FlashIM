"""Mutual dual-softmax matches of a similarity matrix without the ``[L, S]`` probability intermediates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature, platform_for


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

def _eligible(sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None) -> bool:
    return sim.ndim == 2 and sim.dtype == torch.float32 and sim.is_contiguous() and sim.numel() > 0


@ops.dual_softmax_mutual.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _dual_softmax_mutual_impl(
    sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    return dual_softmax_mutual(sim, row_bias, col_bias)
