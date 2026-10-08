# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/fast_fp8_attention.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Batched-head fused FP8 self-attention over static block-32 tensors.

The single Triton kernel consumes a packed
:class:`romav2.full_fp8.QuantizedTensor` with logical shape ``[B, N, 3 * C]``
and emits attention output directly as FP8 under a caller-provided static
output scale exponent. Q/K are dequantized only for high-precision RoPE, then
row-quantized to E4M3 before the QK^T dot; each probability tile is likewise
row-quantized before the PV dot. Both FP8 dots accumulate in FP32, and no
``N x N`` temporary or Python-side per-head loop is used.

Numerical contract
------------------
1. Q/K/V start as E4M3 values with static block-32 E8M0 exponents. Q and K
   are temporarily dequantized to FP32, and optional DINO-style RoPE
   ``(sin, cos)`` is applied in FP32. The first ``prefix = N -
   sin.shape[-2]`` tokens are left untouched; the half-rotation
   ``cat(-x2, x1)`` pairing matches :func:`romav2.vit.attention.rope_apply`.
2. Post-RoPE Q and each K tile are dynamically row-quantized to E4M3 with a
   finite power-of-two scale constant over ``head_dim``. The QK^T dot uses
   those FP8 operands and FP32 accumulation, restoring both row scales before
   applying the softmax scale.
3. Attention probabilities use an FP32 online softmax. Each probability tile
   is row-quantized to E4M3 for the PV dot, whose FP32 accumulation restores
   the probability row scale and the static block-32 V scales. Masked and
   padded rows use finite neutral values, never ``-inf / -inf`` arithmetic.
4. The epilogue normalizes by the FP32 softmax denominator, scales by
   ``2 ** -output_scale_exponent`` per 32-channel block, clamps to the E4M3
   range, and stores FP8 values directly.

The launch grid is ``(ceil(N / BLOCK_M), B * H)`` — exactly one kernel launch
per attention call, tiling across batch * heads and query blocks.
"""

from __future__ import annotations

import torch
from torch import Tensor

from romav2.full_fp8 import QuantizedLayout, QuantizedTensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

__all__ = ["fused_fp8_attention"]


if triton is not None:

    @triton.jit
    def _fp8_attention_kernel(
        qkv_ptr,
        qkv_scale_ptr,
        sin_ptr,
        cos_ptr,
        out_ptr,
        out_scale_ptr,
        H,
        N,
        PREFIX,
        C,
        CP,
        softmax_scale,
        D: tl.constexpr,
        HAS_ROPE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        softmax_scale = tl.full([], softmax_scale, dtype=tl.float32)
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        D_BLOCKS: tl.constexpr = D // 32
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        mask_m = offs_m < N
        d_block = offs_d // 32
        rows_q = b * N + offs_m

        # Static per-32-channel E8M0 scales for this head's slice of Q/K/V and
        # for the output channels [h * D, (h + 1) * D) of the [B, N, C] result.
        q_static_scale = tl.exp2(
            tl.load(qkv_scale_ptr + h * D_BLOCKS + d_block).to(tl.float32)
        )
        k_static_scale = tl.exp2(
            tl.load(qkv_scale_ptr + (C // 32) + h * D_BLOCKS + d_block).to(tl.float32)
        )
        v_scale = tl.exp2(
            tl.load(qkv_scale_ptr + (2 * C) // 32 + h * D_BLOCKS + d_block).to(
                tl.float32
            )
        )
        out_inv_scale = tl.exp2(
            (-tl.load(out_scale_ptr + h * D_BLOCKS + d_block)).to(tl.float32)
        )

        q_base = qkv_ptr + rows_q[:, None] * CP + h * D
        q_bits = tl.load(
            q_base + offs_d[None, :], mask=mask_m[:, None], other=0.0
        ).to(tl.float32)
        q = q_bits * q_static_scale[None, :]

        if HAS_ROPE:
            rot_d = tl.where(offs_d < D // 2, offs_d + D // 2, offs_d - D // 2)
            rot_sign = tl.where(offs_d < D // 2, -1.0, 1.0)
            rot_block = rot_d // 32
            q_rot_scale = tl.exp2(
                tl.load(qkv_scale_ptr + h * D_BLOCKS + rot_block).to(tl.float32)
            )
            k_rot_scale = tl.exp2(
                tl.load(qkv_scale_ptr + (C // 32) + h * D_BLOCKS + rot_block).to(
                    tl.float32
                )
            )
            rope_row_q = tl.maximum(offs_m - PREFIX, 0)
            rope_mask_q = (offs_m >= PREFIX) & mask_m
            q_rot = tl.load(
                q_base + rot_d[None, :], mask=mask_m[:, None], other=0.0
            ).to(tl.float32)
            q_rot = q_rot * q_rot_scale[None, :]
            sin_q = tl.load(
                sin_ptr + rope_row_q[:, None] * D + offs_d[None, :],
                mask=rope_mask_q[:, None],
                other=0.0,
            )
            cos_q = tl.load(
                cos_ptr + rope_row_q[:, None] * D + offs_d[None, :],
                mask=rope_mask_q[:, None],
                other=0.0,
            )
            q = tl.where(
                rope_mask_q[:, None],
                q * cos_q + q_rot * (rot_sign[None, :] * sin_q),
                q,
            )
        # Dynamic, finite power-of-two row scale for post-RoPE Q.
        q_amax = tl.max(tl.abs(q), axis=1)
        q_exp = tl.ceil(tl.log2(q_amax / 448.0))
        q_exp = tl.where(q_amax > 0.0, q_exp, -127.0)
        q_exp = tl.maximum(tl.minimum(q_exp, 127.0), -127.0)
        q_f8 = (q * tl.exp2(-q_exp[:, None])).to(tl.float8e4nv)

        # Keep every online-softmax neutral value in FP32. Scalar Python
        # literals in tl.where can otherwise promote loop-carried values to
        # FP64 during TorchInductor's Triton precompile.
        fp32_zero = tl.full([], 0.0, dtype=tl.float32)
        fp32_one = tl.full([], 1.0, dtype=tl.float32)
        fp32_neg_inf = tl.full([], float("-inf"), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

        if IS_CAUSAL:
            hi = tl.minimum(N, pid_m * BLOCK_M + BLOCK_M)
        else:
            hi = N
        for n0 in range(0, hi, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            rows_k = b * N + offs_n
            k_base = qkv_ptr + rows_k[:, None] * CP + C + h * D
            k_bits = tl.load(
                k_base + offs_d[None, :], mask=mask_n[:, None], other=0.0
            ).to(tl.float32)
            k = k_bits * k_static_scale[None, :]
            if HAS_ROPE:
                rope_row_k = tl.maximum(offs_n - PREFIX, 0)
                rope_mask_k = (offs_n >= PREFIX) & mask_n
                k_rot = tl.load(
                    k_base + rot_d[None, :], mask=mask_n[:, None], other=0.0
                ).to(tl.float32)
                k_rot = k_rot * k_rot_scale[None, :]
                sin_k = tl.load(
                    sin_ptr + rope_row_k[:, None] * D + offs_d[None, :],
                    mask=rope_mask_k[:, None],
                    other=0.0,
                )
                cos_k = tl.load(
                    cos_ptr + rope_row_k[:, None] * D + offs_d[None, :],
                    mask=rope_mask_k[:, None],
                    other=0.0,
                )
                k = tl.where(
                    rope_mask_k[:, None],
                    k * cos_k + k_rot * (rot_sign[None, :] * sin_k),
                    k,
                )

            # Quantize each K row after optional FP32 RoPE, then restore the
            # row scales around an FP32-accumulating FP8 QK^T dot.
            k_amax = tl.max(tl.abs(k), axis=1)
            k_exp = tl.ceil(tl.log2(k_amax / 448.0))
            k_exp = tl.where(k_amax > 0.0, k_exp, -127.0)
            k_exp = tl.maximum(tl.minimum(k_exp, 127.0), -127.0)
            k_f8 = (k * tl.exp2(-k_exp[:, None])).to(tl.float8e4nv)
            scores = tl.dot(
                q_f8,
                tl.trans(k_f8),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            scores = scores * tl.exp2(q_exp[:, None] + k_exp[None, :])
            scores = scores * softmax_scale
            scores = tl.where(
                mask_m[:, None] & mask_n[None, :], scores, fp32_neg_inf
            )
            if IS_CAUSAL:
                scores = tl.where(
                    (offs_n[None, :] <= offs_m[:, None]) & mask_m[:, None],
                    scores,
                    fp32_neg_inf,
                )

            m_new = tl.maximum(m_i, tl.max(scores, 1))
            # Fully masked/padded query rows have m_new=-inf. Replace it
            # before subtraction so online softmax never evaluates -inf/-inf.
            m_new_safe = tl.where(m_new == fp32_neg_inf, fp32_zero, m_new)
            alpha = tl.exp(m_i - m_new_safe)
            p = tl.exp(scores - m_new_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)

            # V remains in its original FP8 representation. Restoring this
            # static block-32 scale after PV is essential to the result.
            v = tl.load(
                qkv_ptr + rows_k[:, None] * CP + 2 * C + h * D + offs_d[None, :],
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float8e4nv)
            p_amax = tl.max(tl.abs(p), axis=1)
            p_exp = tl.ceil(tl.log2(p_amax / 448.0))
            p_exp = tl.where(p_amax > 0.0, p_exp, -127.0)
            p_exp = tl.maximum(tl.minimum(p_exp, 127.0), -127.0)
            p_f8 = (p * tl.exp2(-p_exp[:, None])).to(tl.float8e4nv)
            pv = tl.dot(
                p_f8,
                v,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            pv = pv * tl.exp2(p_exp[:, None]) * v_scale[None, :]
            acc = acc * alpha[:, None] + pv
            m_i = m_new_safe

        l_safe = tl.where(l_i > fp32_zero, l_i, fp32_one)
        acc = acc / l_safe[:, None]
        quantized = tl.maximum(
            tl.minimum(acc * out_inv_scale[None, :], 448.0), -448.0
        ).to(tl.float8e4nv)
        tl.store(
            out_ptr + rows_q[:, None] * C + h * D + offs_d[None, :],
            quantized,
            mask=mask_m[:, None],
        )

else:
    _fp8_attention_kernel = None


def fused_fp8_attention(
    qkv: QuantizedTensor,
    *,
    num_heads: int,
    output_scale_exponent: Tensor,
    rope: tuple[Tensor, Tensor] | None = None,
    is_causal: bool = False,
) -> QuantizedTensor:
    """Run batched-head self-attention on a packed FP8 QKV tensor.

    Args:
        qkv: Quantized tensor with logical shape ``[B, N, 3 * C]`` whose
            channel axis packs Q, K, V back to back. Physical channels may be
            padded past ``3 * C`` to a multiple of 32.
        num_heads: Number of attention heads ``H``; ``C`` must be divisible by
            ``num_heads`` and the head dim must be a power of two divisible by
            32 (production uses 64).
        output_scale_exponent: Static int8 E8M0 exponents of shape
            ``(C // 32,)`` used to quantize the output directly in the kernel.
        rope: Optional ``(sin, cos)`` pair, each ``[N_rope, head_dim]``. The
            first ``N - N_rope`` prefix tokens skip the rotation; RoPE applies
            to Q and K only, in FP32 after dequantization.
        is_causal: Apply a lower-triangular causal mask. The production path
            is non-causal.

    Returns:
        A :class:`QuantizedTensor` with logical shape ``[B, N, C]`` whose FP8
        values are produced by the kernel epilogue and whose scale exponents
        are ``output_scale_exponent``.
    """
    if triton is None or _fp8_attention_kernel is None:
        raise RuntimeError("Fused FP8 attention requires Triton")
    if not isinstance(qkv, QuantizedTensor):
        raise TypeError(f"Expected a QuantizedTensor, got {type(qkv).__name__}")
    if qkv.device.type != "cuda":
        raise RuntimeError("Fused FP8 attention requires CUDA activations")
    if len(qkv.logical_shape) != 3:
        raise ValueError(
            f"Expected qkv logical shape [B, N, 3*C], got {qkv.logical_shape}"
        )
    batch, tokens, three_c = qkv.logical_shape
    if three_c % 3 != 0:
        raise ValueError(f"qkv channels {three_c} are not divisible by 3")
    channels = three_c // 3
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(f"num_heads must be a positive int, got {num_heads}")
    if channels % num_heads != 0:
        raise ValueError(
            f"Channels {channels} are not divisible by num_heads {num_heads}"
        )
    head_dim = channels // num_heads
    if head_dim % 32 != 0 or head_dim & (head_dim - 1) != 0:
        raise ValueError(
            f"head_dim must be a power of two divisible by 32, got {head_dim}"
        )

    output_scale_exponent = output_scale_exponent.to(
        device=qkv.device, dtype=torch.int8
    )
    expected_output_scales = (channels // 32,)
    if tuple(output_scale_exponent.shape) != expected_output_scales:
        raise ValueError(
            f"Expected output scale shape {expected_output_scales}, "
            f"got {tuple(output_scale_exponent.shape)}"
        )
    output_scale_exponent = output_scale_exponent.contiguous()

    has_rope = rope is not None
    if has_rope:
        if not (isinstance(rope, tuple) and len(rope) == 2):
            raise ValueError("rope must be a (sin, cos) tuple")
        sin, cos = rope
        if tuple(sin.shape) != tuple(cos.shape):
            raise ValueError(
                f"rope sin/cos shapes differ: {tuple(sin.shape)} vs {tuple(cos.shape)}"
            )
        if sin.ndim != 2 or sin.shape[-1] != head_dim:
            raise ValueError(
                f"Expected rope shape [N_rope, {head_dim}], got {tuple(sin.shape)}"
            )
        if not (torch.is_floating_point(sin) and torch.is_floating_point(cos)):
            raise ValueError("rope sin/cos must be floating-point tensors")
        if sin.device != qkv.device or cos.device != qkv.device:
            raise ValueError("rope sin/cos must be on the qkv device")
        prefix = tokens - sin.shape[0]
        if prefix < 0:
            raise ValueError(
                f"rope length {sin.shape[0]} exceeds token count {tokens}"
            )
        sin = sin.detach().to(dtype=torch.float32).contiguous()
        cos = cos.detach().to(dtype=torch.float32).contiguous()
    else:
        prefix = tokens
        sin = cos = torch.empty(1, device=qkv.device, dtype=torch.float32)

    matrix = qkv.matrix().contiguous()
    scale_exponent = qkv.scale_exponent.contiguous()
    output = torch.empty(
        batch * tokens, channels, device=qkv.device, dtype=torch.float8_e4m3fn
    )

    # Consumer GPUs expose ~99KB of shared memory per block; larger head dims
    # must drop pipelining stages and narrower K/V tiles to fit.
    if head_dim <= 64:
        block_m, block_n, num_warps, num_stages = 64, 64, 4, 2
    else:
        block_m, block_n, num_warps, num_stages = 64, 32, 4, 1
    grid = (triton.cdiv(tokens, block_m), batch * num_heads)
    _fp8_attention_kernel[grid](
        matrix,
        scale_exponent,
        sin,
        cos,
        output,
        output_scale_exponent,
        num_heads,
        tokens,
        prefix,
        channels,
        matrix.shape[1],
        head_dim**-0.5,
        D=head_dim,
        HAS_ROPE=has_rope,
        IS_CAUSAL=is_causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return QuantizedTensor(
        output.reshape(batch, tokens, channels),
        output_scale_exponent,
        (batch, tokens, channels),
        QuantizedLayout.TOKEN_MATRIX,
    )
