"""Mutual dual-softmax matches of a similarity matrix without the ``[L, S]`` probability intermediates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


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


# 25 tile configurations measured 3.97 to 5.42 ms on LoFTR's 16800 x 16500 matrix (sm_120); these give 4.03 ms.
ROW_TILE = {"block_m": 8, "block_n": 512, "num_warps": 4}
COL_TILE = {"block_m": 64, "block_n": 64, "num_warps": 4}


def dual_softmax_mutual(
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


# -- registrations ---------------------------------------------------------------------------------------

def _eligible(sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None) -> bool:
    return sim.ndim == 2 and sim.dtype == torch.float32 and sim.is_contiguous() and sim.numel() > 0


@ops.dual_softmax_mutual.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _dual_softmax_mutual_impl(
    sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    return dual_softmax_mutual(sim, row_bias, col_bias)
