"""Fused Triton global correlation + positional embedding for Tiny-RoMa (``ops.tiny_roma_corr_pos_embed``).

Computes ``coarse_warp [B, 2, H0, W0]`` directly from ``feat0 [B, C, H0, W0]`` and ``feat1 [B, C, H1, W1]`` by
streaming ``feat1`` tiles in SRAM/L2 against query tiles of ``feat0`` kept in registers, never materializing the
``[B, H1, W1, H0, W0]`` correlation volume (276 MB at 704x1024, 9 GB at 2040x1530).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


@triton.jit
def _corr_pos_embed_kernel(
    q_ptr,
    k_ptr,
    out_ptr,
    n0,
    n1,
    n1_lr,
    H1,
    W1,
    W1_lr,
    scale,
    C: tl.constexpr,
    CC: tl.constexpr,
    EXACT_SOFTMAX: tl.constexpr,
    INPUT_PREC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One ``BLOCK_M`` tile of query pixels from ``feat0`` against all ``n1`` target pixels of ``feat1``."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n0
    offs_c = tl.arange(0, CC)
    mask_c = offs_c < C

    q_base = q_ptr + pid_b.to(tl.int64) * n0 * C
    k_base = k_ptr + pid_b.to(tl.int64) * n1 * C
    q = tl.load(
        q_base + offs_m[:, None].to(tl.int64) * C + offs_c[None, :],
        mask=mask_m[:, None] & mask_c[None, :],
        other=0.0,
    )
    inv_w1 = 1.0 / W1
    inv_h1 = 1.0 / H1
    full_n1 = (n1 // BLOCK_N) * BLOCK_N

    if EXACT_SOFTMAX:
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_y = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for j0 in range(0, n1, BLOCK_N):
            cols = j0 + tl.arange(0, BLOCK_N)
            col_ok = cols < n1
            k = tl.load(
                k_base + cols[:, None].to(tl.int64) * C + offs_c[None, :],
                mask=col_ok[:, None] & mask_c[None, :],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k), input_precision=INPUT_PREC) * scale
            s = tl.where(col_ok[None, :], s, -float("inf"))
            tile_m = tl.max(s, axis=1)
            new_m = tl.maximum(m_i, tile_m)
            alpha = tl.exp(m_i - new_m)
            p = tl.where(col_ok[None, :], tl.exp(s - new_m[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            ky = cols // W1
            kx = cols - ky * W1
            gx = -1.0 + (2.0 * kx.to(tl.float32) + 1.0) * inv_w1
            gy = -1.0 + (2.0 * ky.to(tl.float32) + 1.0) * inv_h1
            acc_x = acc_x * alpha + tl.sum(p * gx[None, :], axis=1)
            acc_y = acc_y * alpha + tl.sum(p * gy[None, :], axis=1)
            m_i = new_m
        out_x = acc_x / l_i
        out_y = acc_y / l_i
    else:
        # Pass 1: exact argmax over all N1 target positions. Since scale > 0, argmax(q @ k^T * scale) == argmax(q @ k^T),
        # saving one [BLOCK_M, BLOCK_N] multiply per iteration; unmasked loop over full tiles saves per-iter masking.
        best_val = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        best_idx = tl.zeros((BLOCK_M,), dtype=tl.int32)
        for j0 in range(0, full_n1, BLOCK_N):
            cols = j0 + tl.arange(0, BLOCK_N)
            k = tl.load(
                k_base + cols[:, None].to(tl.int64) * C + offs_c[None, :],
                mask=mask_c[None, :],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k), input_precision=INPUT_PREC)
            tile_max, tile_idx = tl.max(s, axis=1, return_indices=True)
            better = tile_max > best_val
            best_val = tl.where(better, tile_max, best_val)
            best_idx = tl.where(better, j0 + tile_idx, best_idx)
        if full_n1 < n1:
            cols = full_n1 + tl.arange(0, BLOCK_N)
            col_ok = cols < n1
            k = tl.load(
                k_base + cols[:, None].to(tl.int64) * C + offs_c[None, :],
                mask=col_ok[:, None] & mask_c[None, :],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k), input_precision=INPUT_PREC)
            s = tl.where(col_ok[None, :], s, -float("inf"))
            tile_max, tile_idx = tl.max(s, axis=1, return_indices=True)
            better = tile_max > best_val
            best_val = tl.where(better, tile_max, best_val)
            best_idx = tl.where(better, full_n1 + tile_idx, best_idx)

        # Pass 2: online softmax over the 4x-strided subset (N1 / 16 positions) + the extra `best_idx` logit,
        # gathered directly from the L2-resident `k` buffer (no separate `::4` copy launch).
        by = best_idx // W1
        bx = best_idx - by * W1
        m_i = best_idx.to(tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
        acc_x = -1.0 + (2.0 * bx.to(tl.float32) + 1.0) * inv_w1
        acc_y = -1.0 + (2.0 * by.to(tl.float32) + 1.0) * inv_h1

        for j0 in range(0, n1_lr, BLOCK_N):
            cols = j0 + tl.arange(0, BLOCK_N)
            col_ok = cols < n1_lr
            ly = cols // W1_lr
            lx = cols - ly * W1_lr
            k_row = (ly * 4) * W1 + (lx * 4)
            k_lr = tl.load(
                k_base + k_row[:, None].to(tl.int64) * C + offs_c[None, :],
                mask=col_ok[:, None] & mask_c[None, :],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k_lr), input_precision=INPUT_PREC) * scale
            s = tl.where(col_ok[None, :], s, -float("inf"))
            tile_m = tl.max(s, axis=1)
            new_m = tl.maximum(m_i, tile_m)
            alpha = tl.exp(m_i - new_m)
            p = tl.where(col_ok[None, :], tl.exp(s - new_m[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            gx = -1.0 + (8.0 * lx.to(tl.float32) + 4.0) * inv_w1
            gy = -1.0 + (8.0 * ly.to(tl.float32) + 4.0) * inv_h1
            acc_x = acc_x * alpha + tl.sum(p * gx[None, :], axis=1)
            acc_y = acc_y * alpha + tl.sum(p * gy[None, :], axis=1)
            m_i = new_m
        out_x = acc_x / l_i
        out_y = acc_y / l_i

    out_base = out_ptr + pid_b.to(tl.int64) * 2 * n0
    tl.store(out_base + offs_m, out_x, mask=mask_m)
    tl.store(out_base + n0 + offs_m, out_y, mask=mask_m)


def tiny_roma_corr_pos_embed(
    feat0: Tensor,
    feat1: Tensor,
    exact_softmax: bool = False,
    *,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tensor:
    """Fused Triton correlation + positional embedding; ``(64, 64, 4w, 2s)`` measured fastest on sm_120
    (0.30 ms vs 2.37 ms torch reference, 63% tensor-pipe SOL, 1.2% DRAM, zero ``N1 x N0`` intermediate)."""
    b, c, h0, w0 = feat0.shape
    _, _, h1, w1 = feat1.shape
    n0 = h0 * w0
    n1 = h1 * w1
    # Channels-last inputs permute to contiguous [B, H, W, C] with zero copy.
    q = feat0.permute(0, 2, 3, 1).contiguous().reshape(b, n0, c)
    k = feat1.permute(0, 2, 3, 1).contiguous().reshape(b, n1, c)
    w1_lr = 1 if exact_softmax else w1 // 4
    n1_lr = 0 if exact_softmax else (h1 // 4) * w1_lr
    out = torch.empty((b, 2, h0, w0), device=feat0.device, dtype=torch.float32)
    cc = max(16, triton.next_power_of_2(c))
    grid = (triton.cdiv(n0, block_m), b)
    _corr_pos_embed_kernel[grid](
        q,
        k,
        out,
        n0,
        n1,
        n1_lr,
        h1,
        w1,
        w1_lr,
        1.0 / math.sqrt(c),
        C=c,
        CC=cc,
        EXACT_SOFTMAX=exact_softmax,
        INPUT_PREC="ieee",
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# -- registrations ---------------------------------------------------------------------------------------


def _eligible(feat0: Tensor, feat1: Tensor, exact_softmax: bool = False) -> bool:
    if feat0.ndim != 4 or feat1.ndim != 4:
        return False
    b0, c0, h0, w0 = feat0.shape
    b1, c1, h1, w1 = feat1.shape
    if b0 != b1 or c0 != c1 or c0 < 16 or c0 > 128 or h0 * w0 == 0 or h1 * w1 == 0:
        return False
    if feat0.dtype != feat1.dtype or feat0.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    return exact_softmax or (h1 % 4 == 0 and w1 % 4 == 0)


@ops.tiny_roma_corr_pos_embed.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _corr_pos_embed_impl(feat0: Tensor, feat1: Tensor, exact_softmax: bool = False) -> Tensor:
    return tiny_roma_corr_pos_embed(feat0, feat1, exact_softmax)
