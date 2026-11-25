"""Attention ops: rotary embedding, scaled dot-product attention, multi-scale deformable attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.utils import grid_sample


def _rope_qk_reference(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> tuple[Tensor, Tensor]:
    """Apply half-split RoPE to ``[B, H, N, D]`` q/k with ``[N', D]`` tables.

    q/k are cast to the table dtype, rotated, and cast back. The first
    ``N - N'`` tokens (e.g. CLS/storage tokens) are left untouched.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.to(sin.dtype), k.to(sin.dtype)
    prefix = q.shape[-2] - sin.shape[-2]
    if prefix < 0:
        raise ValueError(f"RoPE table length {sin.shape[-2]} exceeds token count {q.shape[-2]}")

    def apply(value: Tensor) -> Tensor:
        head, tail = value[:, :, :prefix, :], value[:, :, prefix:, :]
        first, second = tail.chunk(2, dim=-1)
        tail = tail * cos + torch.cat((-second, first), dim=-1) * sin
        return torch.cat((head, tail), dim=-2)

    return apply(q).to(q_dtype), apply(k).to(k_dtype)


rope_qk = Op("rope_qk", _rope_qk_reference)


def _apply_rope_2d(value: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply split-half 2D axial RoPE (CroCo convention: ``[y1, y2, x1, x2]``) to ``[B, H, N, D]``."""
    v_dtype = value.dtype
    v = value.to(sin.dtype)
    y, x = v.chunk(2, dim=-1)
    y1, y2 = y.chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-y2, y1, -x2, x1), dim=-1)
    return (v * cos + rotated * sin).to(v_dtype)


def _rope_qk_2d_reference(
    q: Tensor,
    k: Tensor,
    sin: Tensor,
    cos: Tensor,
    k_sin: Tensor | None = None,
    k_cos: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply CroCo-style 2D axial RoPE to ``q [B, H, Nq, D]`` and ``k [B, H, Nk, D]``.

    Each head vector is split into two halves ``(y, x)`` and half-split rotary embedding is applied independently
    to each half (``[y1, y2, x1, x2]`` rotates ``(y1, y2)`` and ``(x1, x2)``). ``sin`` / ``cos`` have shape
    ``[Nq, D]`` (or broadcastable to ``[B, 1, Nq, D]``); for cross-attention where ``k`` has different positions or
    length, pass ``k_sin`` / ``k_cos`` for ``k`` (defaults to ``sin`` / ``cos``).
    """
    if k_sin is None:
        k_sin, k_cos = sin, cos
    return _apply_rope_2d(q, sin, cos), _apply_rope_2d(k, k_sin, k_cos)


rope_qk_2d = Op("rope_qk_2d", _rope_qk_2d_reference)


def _rotate_pairs(x: Tensor) -> Tensor:
    """``(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)``."""
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


def _rope_qk_interleaved_reference(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Interleaved-pair RoPE (LightGlue's convention) in FP32 on ``[B, H, N, D]`` q / k with per-batch tables
    ``[B, 1, N, D]`` (each frequency repeated for its pair): ``t * cos + rotate_pairs(t) * sin``, cast back."""

    def apply(t: Tensor) -> Tensor:
        t32 = t.float()
        return (t32 * cos + _rotate_pairs(t32) * sin).to(t.dtype)

    return apply(q), apply(k)


rope_qk_interleaved = Op("rope_qk_interleaved", _rope_qk_interleaved_reference)


def _qkv_norm_rope_interleaved_reference(
    qkv: Tensor,
    q_weight: Tensor | None,
    q_bias: Tensor | None,
    k_weight: Tensor | None,
    k_bias: Tensor | None,
    eps: float,
    cos: Tensor,
    sin: Tensor,
    num_heads: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Unpack interleaved ``(H, D, 3)`` ``qkv [B, N, 3*H*D]``, apply per-head LayerNorm to Q and K, apply FP32
    interleaved-pair RoPE with ``cos, sin [1|B, N, D]``, and return contiguous ``[B, H, N, D]`` ``(q, k, v)``."""
    q, k, v = qkv.unflatten(-1, (num_heads, -1, 3)).unbind(-1)
    head_dim = q.shape[-1]
    q = torch.nn.functional.layer_norm(q, (head_dim,), q_weight, q_bias, eps).to(v.dtype)
    k = torch.nn.functional.layer_norm(k, (head_dim,), k_weight, k_bias, eps).to(v.dtype)
    q_hnd = q.transpose(1, 2)
    k_hnd = k.transpose(1, 2)
    v_hnd = v.transpose(1, 2).contiguous()
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    q_rot = (q_hnd.float() * cos_b.float() + _rotate_pairs(q_hnd.float()) * sin_b.float()).to(v.dtype)
    k_rot = (k_hnd.float() * cos_b.float() + _rotate_pairs(k_hnd.float()) * sin_b.float()).to(v.dtype)
    return q_rot.contiguous(), k_rot.contiguous(), v_hnd


qkv_norm_rope_interleaved = Op("qkv_norm_rope_interleaved", _qkv_norm_rope_interleaved_reference)


def _qk_norm_swap_heads_reference(
    qk: Tensor,
    v: Tensor,
    qk_weight: Tensor | None,
    qk_bias: Tensor | None,
    eps: float,
    num_heads: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-head LayerNorm on ``qk [2B, N, H*D]`` and batch-half swap for stacked cross-attention: returns contiguous
    ``[2B, H, N, D]`` ``(q, swap(q), swap(v))`` where ``swap(t) = cat([t[B:], t[:B]])``."""
    half = qk.shape[0] // 2
    qk_h = qk.unflatten(-1, (num_heads, -1))
    v_h = v.unflatten(-1, (num_heads, -1))
    q = torch.nn.functional.layer_norm(qk_h, (qk_h.shape[-1],), qk_weight, qk_bias, eps).to(v.dtype)
    q_hnd = q.transpose(1, 2).contiguous()
    v_hnd = v_h.transpose(1, 2)
    k_swap = torch.cat((q_hnd[half:], q_hnd[:half]), dim=0).contiguous()
    v_swap = torch.cat((v_hnd[half:], v_hnd[:half]), dim=0).contiguous()
    return q_hnd, k_swap, v_swap


qk_norm_swap_heads = Op("qk_norm_swap_heads", _qk_norm_swap_heads_reference)

# -- scaled dot-product attention (non-causal, no mask) ------------------------------------------------


def _sdpa_reference(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """``softmax(q k^T / sqrt(D)) v`` for ``[B, H, N, D]`` tensors (torch SDPA)."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


sdpa = Op("sdpa", _sdpa_reference)


# -- shared-QK bidirectional cross-attention --------------------------------------------------------------


def _bidirectional_cross_attention_reference(
    qk0: Tensor, qk1: Tensor, v0: Tensor, v1: Tensor, num_heads: int
) -> tuple[Tensor, Tensor]:
    """Shared-QK bidirectional cross-attention on ``[M, L0, C]`` / ``[M, L1, C]`` (FP32 softmax).

    Computes ``sim = (qk0 @ qk1^T) / sqrt(D)`` per head, then ``out0 = softmax(sim, -1) @ v1`` and
    ``out1 = softmax(sim^T, -1) @ v0``, returning ``(out0 [M, L0, C], out1 [M, L1, C])`` in ``v0``'s dtype."""
    bs, _, channels = qk0.shape
    if bs == 0:
        return torch.empty_like(v0), torch.empty_like(v1)
    head_dim = channels // num_heads
    q0, q1, val0, val1 = (
        t.reshape(bs, t.shape[1], num_heads, head_dim).permute(0, 2, 1, 3) for t in (qk0, qk1, v0, v1)
    )
    sim = (q0.float() @ q1.float().transpose(-2, -1)) * (head_dim**-0.5)
    attn01 = torch.softmax(sim, dim=-1).to(val1.dtype)
    attn10 = torch.softmax(sim.transpose(-2, -1), dim=-1).to(val0.dtype)
    out0 = (attn01 @ val1).transpose(1, 2).flatten(-2)
    out1 = (attn10 @ val0).transpose(1, 2).flatten(-2)
    return out0, out1


bidirectional_cross_attention = Op("bidirectional_cross_attention", _bidirectional_cross_attention_reference)


# -- ELU+1 linear attention -------------------------------------------------------------------------------


def _elu_linear_attention_reference(
    q: Tensor, k: Tensor, v: Tensor, eps: float = 1e-6, gemm_dtype: torch.dtype | None = None
) -> Tensor:
    """ELU+1 kernel linear attention on ``q [B, L, H, D]``, ``k [B, S, H, D]``, ``v [B, S, H, V]`` -> ``[B, L, H, V]``
    in ``q.dtype``. Values are scaled by ``1 / S`` before accumulating ``K^T V`` and multiplied back by ``S`` on output
    (LoFTR overflow guard). ``gemm_dtype`` runs the two contractions on operands cast to that dtype while ELU,
    normalizer and scaling stay FP32."""
    if q.shape[1] == 0 or k.shape[1] == 0:
        return q.new_zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]))
    qf, kf, vf = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0, v.float()
    length = vf.shape[1]
    z = 1.0 / (torch.einsum("nlhd,nhd->nlh", qf, kf.sum(dim=1)) + eps)
    if gemm_dtype is None:
        kv = torch.einsum("nshd,nshv->nhdv", kf, vf / length)
        out = torch.einsum("nlhd,nhdv,nlh->nlhv", qf, kv, z) * length
    else:
        kv = torch.einsum("nshd,nshv->nhdv", kf.to(gemm_dtype), (vf / length).to(gemm_dtype))
        out = torch.einsum("nlhd,nhdv->nlhv", qf.to(gemm_dtype), kv).float() * z[..., None] * length
    return out.to(q.dtype)


elu_linear_attention = Op("elu_linear_attention", _elu_linear_attention_reference)


# -- multi-scale deformable attention (Deformable DETR core) --------------------------------------------

# Reference sampling tensors are built per query chunk of about this many float32 elements.
_DEFORM_CHUNK_ELEMENTS = 1 << 26


def _deformable_attention_reference(
    value: Tensor,
    offsets: Tensor,
    weights: Tensor,
    reference: Tensor,
    spatial_shapes: tuple[tuple[int, int], ...],
    softmax: bool = False,
) -> Tensor:
    """Multi-scale deformable attention with bilinear sampling (zero padding, ``align_corners=False``).

    ``value`` ``[N, S, M, D]`` is the flattened multi-level feature map (``S = sum H_l * W_l``, levels in
    ``spatial_shapes`` order, row-major); ``offsets`` ``[N, Lq, M, L, P, 2]`` are sampling offsets in
    level pixels ``(x, y)``; ``weights`` ``[N, Lq, M, L, P]`` are attention weights, or logits softmaxed (FP32,
    over each head's ``L * P`` points) when ``softmax``; ``reference`` ``[N, Lq, 2]`` is each query's
    normalized ``(x, y)`` reference point in ``[0, 1]``, shared by every level. Returns ``[N, Lq, M * D]`` in
    ``value``'s dtype. Sampling runs in float32 and is done in query chunks to bound memory.
    """
    n, _, m, d = value.shape
    lq, levels, points = offsets.shape[1], offsets.shape[3], offsets.shape[4]
    if len(spatial_shapes) != levels:
        raise ValueError(f"{len(spatial_shapes)} spatial shapes for {levels} levels")
    if softmax:
        weights = torch.softmax(weights.float().reshape(n, lq, m, levels * points), -1).view(n, lq, m, levels, points)
    maps = [
        chunk.permute(0, 2, 3, 1).reshape(n * m, d, h, w).float()
        for chunk, (h, w) in zip(value.split([h * w for h, w in spatial_shapes], dim=1), spatial_shapes)
    ]
    chunk_queries = max(1, _DEFORM_CHUNK_ELEMENTS // (n * m * d * points))
    outputs = []
    for start in range(0, lq, chunk_queries):
        stop = min(start + chunk_queries, lq)
        offs = offsets[:, start:stop].float()
        ref = reference[:, start:stop].float()[:, None, :, None, :]  # [N, 1, Lc, 1, 2]
        wts = weights[:, start:stop].float().permute(0, 2, 1, 3, 4)  # [N, M, Lc, L, P]
        acc = None
        for level, ((h, w), fmap) in enumerate(zip(spatial_shapes, maps)):
            normalizer = offs.new_tensor([w, h])
            loc = ref + offs[:, :, :, level].permute(0, 2, 1, 3, 4).reshape(n, m, stop - start, points, 2) / normalizer
            grid = (2 * loc - 1).reshape(n * m, stop - start, points, 2)
            sampled = grid_sample(
                fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # [N*M, D, Lc, P]
            term = (sampled * wts[:, :, :, level].reshape(n * m, 1, stop - start, points)).sum(-1)
            acc = term if acc is None else acc + term
        outputs.append(acc.reshape(n, m * d, stop - start).transpose(1, 2))
    return torch.cat(outputs, dim=1).to(value.dtype).contiguous()


deformable_attention = Op("deformable_attention", _deformable_attention_reference)
