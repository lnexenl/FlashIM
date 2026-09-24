"""One-launch RoPE for Q and K (ported from RoMaV2 ``rope_fusion.py``)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_qk_kernel(
    q_ptr,
    k_ptr,
    sin_ptr,
    cos_ptr,
    q_out_ptr,
    k_out_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_n,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_n,
    k_stride_d,
    sin_stride_n,
    sin_stride_d,
    cos_stride_n,
    cos_stride_d,
    H,
    N,
    D,
    PREFIX,
    BLOCK_D: tl.constexpr,
    TABLES_FP32: tl.constexpr,
    TABLES_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    b = row // (H * N)
    hn = row % (H * N)
    h = hn // N
    n = hn % N
    mask = d < D

    q_offset = b * q_stride_b + h * q_stride_h + n * q_stride_n + d * q_stride_d
    k_offset = b * k_stride_b + h * k_stride_h + n * k_stride_n + d * k_stride_d
    q_value = tl.load(q_ptr + q_offset, mask=mask, other=0.0)
    k_value = tl.load(k_ptr + k_offset, mask=mask, other=0.0)

    rope_n = tl.maximum(n - PREFIX, 0)
    rotated_d = tl.where(d < D // 2, d + D // 2, d - D // 2)
    q_rotated = tl.load(
        q_ptr + b * q_stride_b + h * q_stride_h + n * q_stride_n + rotated_d * q_stride_d,
        mask=mask,
        other=0.0,
    )
    k_rotated = tl.load(
        k_ptr + b * k_stride_b + h * k_stride_h + n * k_stride_n + rotated_d * k_stride_d,
        mask=mask,
        other=0.0,
    )
    sin = tl.load(sin_ptr + rope_n * sin_stride_n + d * sin_stride_d, mask=mask, other=0.0)
    cos = tl.load(cos_ptr + rope_n * cos_stride_n + d * cos_stride_d, mask=mask, other=0.0)
    q_rotated_term = q_rotated * sin
    k_rotated_term = k_rotated * sin
    q_rotated_term = tl.where(d < D // 2, -q_rotated_term, q_rotated_term)
    k_rotated_term = tl.where(d < D // 2, -k_rotated_term, k_rotated_term)
    if TABLES_FP32:
        q_roped = q_value * cos + q_rotated_term
        k_roped = k_value * cos + k_rotated_term
    elif TABLES_BF16:
        q_scaled = (q_value * cos).to(tl.bfloat16)
        k_scaled = (k_value * cos).to(tl.bfloat16)
        q_rotated_term = q_rotated_term.to(tl.bfloat16)
        k_rotated_term = k_rotated_term.to(tl.bfloat16)
        q_roped = (q_scaled + q_rotated_term).to(tl.bfloat16)
        k_roped = (k_scaled + k_rotated_term).to(tl.bfloat16)
    else:
        q_scaled = (q_value * cos).to(tl.float16)
        k_scaled = (k_value * cos).to(tl.float16)
        q_rotated_term = q_rotated_term.to(tl.float16)
        k_rotated_term = k_rotated_term.to(tl.float16)
        q_roped = (q_scaled + q_rotated_term).to(tl.float16)
        k_roped = (k_scaled + k_rotated_term).to(tl.float16)
    q_value = tl.where(n >= PREFIX, q_roped, q_value)
    k_value = tl.where(n >= PREFIX, k_roped, k_value)

    out_offset = ((b * H + h) * N + n) * D + d
    tl.store(q_out_ptr + out_offset, q_value, mask=mask)
    tl.store(k_out_ptr + out_offset, k_value, mask=mask)


def rope_qk(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> tuple[Tensor, Tensor]:
    batch, heads, tokens, head_dim = q.shape
    q_out = torch.empty((batch, heads, tokens, head_dim), device=q.device, dtype=q.dtype)
    k_out = torch.empty((batch, heads, tokens, head_dim), device=k.device, dtype=k.dtype)
    _rope_qk_kernel[(batch * heads * tokens,)](
        q,
        k,
        sin,
        cos,
        q_out,
        k_out,
        *q.stride(),
        *k.stride(),
        *sin.stride(),
        *cos.stride(),
        heads,
        tokens,
        head_dim,
        tokens - sin.shape[0],
        BLOCK_D=triton.next_power_of_2(head_dim),
        TABLES_FP32=sin.dtype == torch.float32,
        TABLES_BF16=sin.dtype == torch.bfloat16,
        num_warps=4,
    )
    return q_out, k_out
