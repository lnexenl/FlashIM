"""Triton kernels for descriptor distance and ratio-test mutual-nearest-neighbour matching.

``hamming_distances``: packed ``int32`` XOR + hardware ``libdevice.popc`` (eliminates the 4.3 GB int64 LUT in ORB).
``hamming_ratio_mutual_match``: fused tiled XOR + ``popc`` + row top-2 + column argmin without materializing ``[N, M]``
or running radix-sort ``topk`` (only 1 host sync for the final match count).
``ratio_mutual_match``: single-pass tiled row top-2 + column argmin over a non-negative FP32 ``[N, M]`` distance matrix,
replacing PyTorch's multi-block radix sort (``mbtopk``) and separate ``argmin`` pass.
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

_INT32_MAX = tl.constexpr(0x7FFFFFFF)
_FP32_INF_BITS = tl.constexpr(0x7F800000)
_COL_INIT = (0x7FFFFFFF << 32) | 0xFFFFFFFF
_COL_FP32_INIT = (0x7F800000 << 32) | 0xFFFFFFFF


@triton.jit
def _hamming_distances_kernel(
    a_ptr, b_ptr, out_ptr, N, M,
    WORDS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.int32)
    for w in tl.static_range(WORDS):
        av = tl.load(a_ptr + offs_n.to(tl.int64) * WORDS + w, mask=mask_n, other=0)
        bv = tl.load(b_ptr + offs_m.to(tl.int64) * WORDS + w, mask=mask_m, other=0)
        acc += libdevice.popc(av[:, None] ^ bv[None, :])

    out_ptrs = out_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_n[:, None] & mask_m[None, :])


def _hamming_distances_triton(a: Tensor, b: Tensor) -> Tensor:
    n, d = a.shape
    m = b.shape[0]
    if n == 0 or m == 0:
        return a.new_empty(n, m, dtype=torch.float32)
    a32 = a.contiguous().view(torch.int32)
    b32 = b.contiguous().view(torch.int32)
    words = d // 4
    out = torch.empty(n, m, device=a.device, dtype=torch.float32)
    block_n, block_m = 64, 64
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
    _hamming_distances_kernel[grid](
        a32, b32, out, n, m, WORDS=words, BLOCK_N=block_n, BLOCK_M=block_m, num_warps=4,
    )
    return out


def _hamming_eligible(a: Tensor, b: Tensor, *_) -> bool:
    return (
        a.is_cuda and b.is_cuda and a.dtype == torch.uint8 and b.dtype == torch.uint8
        and a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1]
        and a.shape[1] > 0 and a.shape[1] % 4 == 0 and a.shape[1] <= 128
    )


ops.hamming_distances.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_hamming_eligible
)(_hamming_distances_triton)


@triton.jit
def _hamming_top2_colmin_kernel(
    a_ptr, b_ptr, d1_ptr, d2_ptr, j1_ptr, col_best_ptr, N, M,
    WORDS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """One program per ``BLOCK_N`` rows: streams all ``M`` columns in ``BLOCK_M`` tiles, updating exact row top-2
    ``(d1, d2, best_j)`` in registers and column argmin via ``atomic_min`` on packed ``(dist << 32) | row_idx``
    (smallest row index wins ties, matching ``torch.argmin(dim=0)``)."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    row_i64 = offs_n.to(tl.int64)

    best1 = tl.full((BLOCK_N,), _INT32_MAX, dtype=tl.int32)
    best2 = tl.full((BLOCK_N,), _INT32_MAX, dtype=tl.int32)
    best1_j = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.int32)
        for w in tl.static_range(WORDS):
            av = tl.load(a_ptr + row_i64 * WORDS + w, mask=mask_n, other=0)
            bv = tl.load(b_ptr + offs_m.to(tl.int64) * WORDS + w, mask=mask_m, other=0)
            acc += libdevice.popc(av[:, None] ^ bv[None, :])

        d_row = tl.where(mask_m[None, :], acc, _INT32_MAX)
        d_col = tl.where(mask_n[:, None] & mask_m[None, :], acc, _INT32_MAX)

        packed_col = (d_col.to(tl.int64) << 32) | row_i64[:, None]
        tile_col_min = tl.min(packed_col, axis=0)
        tl.atomic_min(col_best_ptr + offs_m, tile_col_min, mask=mask_m)

        packed_row = (d_row.to(tl.int64) << 32) | offs_m[None, :].to(tl.int64)
        m1 = tl.min(packed_row, axis=1)
        t_d1 = (m1 >> 32).to(tl.int32)
        t_j1 = (m1 & 0xFFFFFFFF).to(tl.int32)
        d_no1 = tl.where(offs_m[None, :] == t_j1[:, None], _INT32_MAX, d_row)
        t_d2 = tl.min(d_no1, axis=1)

        better = t_d1 < best1
        best2 = tl.where(better, tl.minimum(best1, t_d2), tl.minimum(best2, t_d1))
        best1_j = tl.where(better, t_j1, best1_j)
        best1 = tl.where(better, t_d1, best1)

    tl.store(d1_ptr + offs_n, best1, mask=mask_n)
    tl.store(d2_ptr + offs_n, best2, mask=mask_n)
    tl.store(j1_ptr + offs_n, best1_j, mask=mask_n)


def _hamming_ratio_mutual_match_triton(a: Tensor, b: Tensor, ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    n, d = a.shape
    m = b.shape[0]
    if n == 0 or m < 2:
        empty = torch.empty(0, dtype=torch.long, device=a.device)
        return empty, empty, a.new_empty(0, dtype=torch.float32)
    a32 = a.contiguous().view(torch.int32)
    b32 = b.contiguous().view(torch.int32)
    words = d // 4
    d1 = torch.empty(n, device=a.device, dtype=torch.int32)
    d2 = torch.empty(n, device=a.device, dtype=torch.int32)
    j1 = torch.empty(n, device=a.device, dtype=torch.int32)
    col_best = torch.full((m,), _COL_INIT, device=a.device, dtype=torch.int64)

    block_n, block_m = 32, 64
    grid = (triton.cdiv(n, block_n),)
    _hamming_top2_colmin_kernel[grid](
        a32, b32, d1, d2, j1, col_best, n, m, WORDS=words, BLOCK_N=block_n, BLOCK_M=block_m, num_warps=4,
    )
    col_argmin = (col_best & 0xFFFFFFFF).to(torch.long)
    best_j = j1.to(torch.long)
    d1_f, d2_f = d1.float(), d2.float()
    ids = torch.arange(n, device=a.device)
    keep = (d1_f < ratio * d2_f) & (col_argmin[best_j] == ids)
    ia = keep.nonzero()[:, 0]
    return ia, best_j[ia], 1.0 - d1_f[ia] / d2_f[ia]


ops.hamming_ratio_mutual_match.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_hamming_eligible
)(_hamming_ratio_mutual_match_triton)


@triton.jit
def _ratio_top2_colmin_fp32_kernel(
    dist_ptr, d1_ptr, d2_ptr, j1_ptr, col_best_ptr, N, M, stride_n, stride_m,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Row top-2 + column argmin over a non-negative FP32 ``[N, M]`` distance matrix. For non-negative IEEE-754
    floats, the uint32 bit representation is strictly order-preserving (after mapping ``-0.0`` to ``+0.0``), so
    packing ``(bits << 32) | idx`` into signed int64 (always non-negative since sign bit 31 is 0) gives exact
    smallest-index tie-breaking via ``tl.min`` / ``tl.atomic_min``."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    row_i64 = offs_n.to(tl.int64)

    best1 = tl.full((BLOCK_N,), _FP32_INF_BITS, dtype=tl.int32)
    best2 = tl.full((BLOCK_N,), _FP32_INF_BITS, dtype=tl.int32)
    best1_j = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        valid = mask_n[:, None] & mask_m[None, :]
        ptrs = dist_ptr + row_i64[:, None] * stride_n + offs_m[None, :].to(tl.int64) * stride_m
        val = tl.load(ptrs, mask=valid, other=float("inf")).to(tl.float32)
        bits = tl.where(val == 0.0, 0, val.to(tl.int32, bitcast=True))
        bits_row = tl.where(mask_m[None, :], bits, _FP32_INF_BITS)
        bits_col = tl.where(valid, bits, _FP32_INF_BITS)

        packed_col = (bits_col.to(tl.int64) << 32) | row_i64[:, None]
        tile_col_min = tl.min(packed_col, axis=0)
        tl.atomic_min(col_best_ptr + offs_m, tile_col_min, mask=mask_m)

        packed_row = (bits_row.to(tl.int64) << 32) | offs_m[None, :].to(tl.int64)
        m1 = tl.min(packed_row, axis=1)
        t_b1 = (m1 >> 32).to(tl.int32)
        t_j1 = (m1 & 0xFFFFFFFF).to(tl.int32)
        b_no1 = tl.where(offs_m[None, :] == t_j1[:, None], _FP32_INF_BITS, bits_row)
        t_b2 = tl.min(b_no1, axis=1)

        better = t_b1 < best1
        best2 = tl.where(better, tl.minimum(best1, t_b2), tl.minimum(best2, t_b1))
        best1_j = tl.where(better, t_j1, best1_j)
        best1 = tl.where(better, t_b1, best1)

    tl.store(d1_ptr + offs_n, best1.to(tl.float32, bitcast=True), mask=mask_n)
    tl.store(d2_ptr + offs_n, best2.to(tl.float32, bitcast=True), mask=mask_n)
    tl.store(j1_ptr + offs_n, best1_j, mask=mask_n)


def _ratio_mutual_match_triton(dist: Tensor, ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    n, m = dist.shape
    if n == 0 or m < 2:
        empty = torch.empty(0, dtype=torch.long, device=dist.device)
        return empty, empty, dist.new_empty(0, dtype=torch.float32)
    d1 = torch.empty(n, device=dist.device, dtype=torch.float32)
    d2 = torch.empty(n, device=dist.device, dtype=torch.float32)
    j1 = torch.empty(n, device=dist.device, dtype=torch.int32)
    col_best = torch.full((m,), _COL_FP32_INIT, device=dist.device, dtype=torch.int64)

    block_n, block_m = 32, 64
    grid = (triton.cdiv(n, block_n),)
    _ratio_top2_colmin_fp32_kernel[grid](
        dist, d1, d2, j1, col_best, n, m, dist.stride(0), dist.stride(1),
        BLOCK_N=block_n, BLOCK_M=block_m, num_warps=4,
    )
    col_argmin = (col_best & 0xFFFFFFFF).to(torch.long)
    best_j = j1.to(torch.long)
    ids = torch.arange(n, device=dist.device)
    keep = (d1 < ratio * d2) & (col_argmin[best_j] == ids)
    ia = keep.nonzero()[:, 0]
    return ia, best_j[ia], 1.0 - d1[ia] / d2[ia]


def _ratio_eligible(dist: Tensor, ratio: float) -> bool:
    return dist.is_cuda and dist.ndim == 2 and dist.dtype == torch.float32


ops.ratio_mutual_match.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_ratio_eligible
)(_ratio_mutual_match_triton)
