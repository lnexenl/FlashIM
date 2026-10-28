"""Attention ops: rotary embedding, scaled dot-product attention, multi-scale deformable attention."""

from __future__ import annotations

import torch
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

# -- scaled dot-product attention (non-causal, no mask) ------------------------------------------------


def _sdpa_reference(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """``softmax(q k^T / sqrt(D)) v`` for ``[B, H, N, D]`` tensors (torch SDPA)."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


sdpa = Op("sdpa", _sdpa_reference)


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
