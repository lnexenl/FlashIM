# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/rope_fusion.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Inference-only fused rotary-position preparation for local attention.

The Triton path consumes transposed Q/K views and writes compact ``[B, H, N, D]``
outputs directly.  Training and unsupported layouts retain the eager expression so
autograd and the original dtype-promotion semantics remain intact.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

__all__ = ["fused_rope_apply_qk"]


_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


if triton is not None:

    @triton.jit
    def _fused_rope_apply_qk_kernel(
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

        q_offset = (
            b * q_stride_b + h * q_stride_h + n * q_stride_n + d * q_stride_d
        )
        k_offset = (
            b * k_stride_b + h * k_stride_h + n * k_stride_n + d * k_stride_d
        )
        q_value = tl.load(q_ptr + q_offset, mask=mask, other=0.0)
        k_value = tl.load(k_ptr + k_offset, mask=mask, other=0.0)

        rope_n = tl.maximum(n - PREFIX, 0)
        rotated_d = tl.where(d < D // 2, d + D // 2, d - D // 2)
        q_rotated = tl.load(
            q_ptr
            + b * q_stride_b
            + h * q_stride_h
            + n * q_stride_n
            + rotated_d * q_stride_d,
            mask=mask,
            other=0.0,
        )
        k_rotated = tl.load(
            k_ptr
            + b * k_stride_b
            + h * k_stride_h
            + n * k_stride_n
            + rotated_d * k_stride_d,
            mask=mask,
            other=0.0,
        )
        sin = tl.load(
            sin_ptr + rope_n * sin_stride_n + d * sin_stride_d,
            mask=mask,
            other=0.0,
        )
        cos = tl.load(
            cos_ptr + rope_n * cos_stride_n + d * cos_stride_d,
            mask=mask,
            other=0.0,
        )
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

else:
    _fused_rope_apply_qk_kernel = None


def _eager_rope_apply_qk(
    q: Tensor, k: Tensor, sin: Tensor, cos: Tensor
) -> tuple[Tensor, Tensor]:
    """Keep the original local-attention RoPE expression and autograd graph."""
    q_dtype, k_dtype = q.dtype, k.dtype
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    prefix = q.shape[-2] - sin.shape[-2]
    assert prefix >= 0

    def apply(value: Tensor) -> Tensor:
        value_prefix = value[:, :, :prefix, :]
        value_suffix = value[:, :, prefix:, :]
        first, second = value_suffix.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        value_suffix = (value_suffix * cos) + (rotated * sin)
        return torch.cat((value_prefix, value_suffix), dim=-2)

    return apply(q).to(dtype=q_dtype), apply(k).to(dtype=k_dtype)


def _triton_ready(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> bool:
    """Return whether the one-launch path can preserve local-attention math."""
    if triton is None or torch.is_grad_enabled() or torch.onnx.is_in_onnx_export():
        return False
    if not (
        q.is_cuda
        and k.is_cuda
        and sin.is_cuda
        and cos.is_cuda
        and q.device == k.device == sin.device == cos.device
    ):
        return False
    device_index = (
        q.device.index if q.device.index is not None else torch.cuda.current_device()
    )
    if torch.cuda.get_device_capability(device_index) < (12, 0):
        return False
    if q.dtype not in _TRITON_DTYPES or k.dtype not in _TRITON_DTYPES:
        return False
    if sin.dtype != cos.dtype or sin.dtype not in _TRITON_DTYPES:
        return False
    # FP32 tables naturally promote any supported Q/K dtype.  For lower
    # precision tables, local attention casts Q/K to the table dtype first.
    if sin.dtype != torch.float32 and (q.dtype != sin.dtype or k.dtype != sin.dtype):
        return False
    if q.ndim != 4 or k.shape != q.shape or sin.shape != cos.shape or sin.ndim != 2:
        return False
    batch, heads, tokens, head_dim = q.shape
    return (
        batch >= 1
        and heads >= 1
        and sin.shape[0] > 0
        and tokens >= sin.shape[0]
        and head_dim == sin.shape[1]
        and head_dim > 0
        and head_dim % 2 == 0
        and head_dim <= 1024
    )


def fused_rope_apply_qk(
    q: Tensor, k: Tensor, sin: Tensor, cos: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply local-attention RoPE to Q/K in one Triton launch when eligible.

    The fused path is intentionally limited to CUDA evaluation under disabled
    gradients.  It reads arbitrary Q/K source strides (including the
    ``transpose(1, 2)`` views produced by local attention) and emits contiguous
    outputs.  Unsupported devices, dtypes, layouts, and autograd executions
    use the eager implementation with its original promotion behavior.
    """
    if not _triton_ready(q, k, sin, cos):
        return _eager_rope_apply_qk(q, k, sin, cos)

    batch, heads, tokens, head_dim = q.shape
    prefix = tokens - sin.shape[0]
    q_out = torch.empty(
        (batch, heads, tokens, head_dim), device=q.device, dtype=q.dtype
    )
    k_out = torch.empty(
        (batch, heads, tokens, head_dim), device=k.device, dtype=k.dtype
    )
    block_d = triton.next_power_of_2(head_dim)
    _fused_rope_apply_qk_kernel[(batch * heads * tokens,)](
        q,
        k,
        sin,
        cos,
        q_out,
        k_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        sin.stride(0),
        sin.stride(1),
        cos.stride(0),
        cos.stride(1),
        heads,
        tokens,
        head_dim,
        prefix,
        BLOCK_D=block_d,
        TABLES_FP32=sin.dtype == torch.float32,
        TABLES_BF16=sin.dtype == torch.bfloat16,
        num_warps=4,
    )
    return q_out, k_out
