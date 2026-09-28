"""Attention ops: rotary embedding, scaled dot-product attention, multi-scale deformable attention."""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op
from flashim.platforms import Feature


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

_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


def _triton_eligible(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> bool:
    if torch.is_grad_enabled() or q.ndim != 4 or k.shape != q.shape:
        return False
    if sin.ndim != 2 or sin.shape != cos.shape or sin.dtype != cos.dtype:
        return False
    if q.dtype not in _FLOATS or k.dtype not in _FLOATS or sin.dtype not in _FLOATS:
        return False
    # Low-precision tables require q/k already in the table dtype (no promotion).
    if sin.dtype != torch.float32 and (q.dtype != sin.dtype or k.dtype != sin.dtype):
        return False
    head_dim = q.shape[-1]
    return 0 < sin.shape[0] <= q.shape[2] and head_dim == sin.shape[1] and head_dim % 2 == 0


@rope_qk.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_triton_eligible)
def _rope_qk_triton(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> tuple[Tensor, Tensor]:
    from flashim.ops._triton.attention import rope_qk as kernel

    return kernel(q, k, sin, cos)


# -- scaled dot-product attention (non-causal, no mask) ------------------------------------------------


def _sdpa_reference(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """``softmax(q k^T / sqrt(D)) v`` for ``[B, H, N, D]`` tensors (torch SDPA)."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


sdpa = Op("sdpa", _sdpa_reference)


def _sage_eligible(q: Tensor, k: Tensor, v: Tensor) -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("sageattention") is not None
        and q.ndim == 4
        and q.shape == k.shape == v.shape
        and q.shape[-1] in (64, 128)
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    )


# SageAttention (thu-ml; Q/K INT8 with K smoothing). Opt-in: pin with ``ops.sdpa.get("sage")`` (INT8 QK +
# FP16 PV, FP32 accumulation) or ``"sage-fp8"`` (INT8 QK + FP8 PV).
@sdpa.register("sage", device=DeviceSpec("cuda", min_capability=(8, 0)), priority=-1, when=_sage_eligible)
def _sdpa_sage(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return _sage_sdpa(q, k, v, "fp16")


@sdpa.register("sage-fp8", device=DeviceSpec("cuda", min_capability=(8, 9)), priority=-1, when=_sage_eligible)
def _sdpa_sage_fp8(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return _sage_sdpa(q, k, v, "fp8")


@torch.library.custom_op("flashim::sage_sdpa", mutates_args=())
def _sage_sdpa(q: Tensor, k: Tensor, v: Tensor, variant: str) -> Tensor:
    import sageattention

    if variant == "fp8":
        return sageattention.sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, tensor_layout="HND", is_causal=False, pv_accum_dtype="fp32+fp32"
        )
    return sageattention.sageattn_qk_int8_pv_fp16_cuda(
        q, k, v, tensor_layout="HND", is_causal=False, pv_accum_dtype="fp32"
    )


@_sage_sdpa.register_fake
def _(q: Tensor, k: Tensor, v: Tensor, variant: str) -> Tensor:
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


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
            sampled = torch.nn.functional.grid_sample(
                fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # [N*M, D, Lc, P]
            term = (sampled * wts[:, :, :, level].reshape(n * m, 1, stop - start, points)).sum(-1)
            acc = term if acc is None else acc + term
        outputs.append(acc.reshape(n, m * d, stop - start).transpose(1, 2))
    return torch.cat(outputs, dim=1).to(value.dtype).contiguous()


deformable_attention = Op("deformable_attention", _deformable_attention_reference)


def _deformable_triton_eligible(value, offsets, weights, reference, spatial_shapes, softmax=False) -> bool:
    if torch.is_grad_enabled() and any(t.requires_grad for t in (value, offsets, weights, reference)):
        return False  # forward-only kernel
    if value.ndim != 4 or offsets.ndim != 6 or weights.ndim != 5:
        return False
    return (
        value.dtype in _FLOATS
        and value.shape[-1] in (8, 16, 32, 64, 128)
        and offsets.shape[3] == len(spatial_shapes)
        and value.shape[1] == sum(h * w for h, w in spatial_shapes)
        and offsets[0, 0].is_contiguous()
        and weights[0, 0].is_contiguous()
    )


@deformable_attention.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_deformable_triton_eligible
)
def _deformable_attention_triton(value, offsets, weights, reference, spatial_shapes, softmax=False) -> Tensor:
    flat = [size for shape in spatial_shapes for size in shape]
    return _deformable_attention_kernel(value, offsets, weights, reference, flat, softmax)


# Opaque to torch.compile (no tracing into the launcher; strided offset/weight views pass through as-is).
@torch.library.custom_op("flashim::deformable_attention", mutates_args=())
def _deformable_attention_kernel(
    value: Tensor, offsets: Tensor, weights: Tensor, reference: Tensor, shapes: list[int], softmax: bool
) -> Tensor:
    from flashim.ops._triton.attention import deformable_attention as kernel

    spatial_shapes = tuple((shapes[i], shapes[i + 1]) for i in range(0, len(shapes), 2))
    return kernel(value, offsets, weights, reference, spatial_shapes, softmax)


@_deformable_attention_kernel.register_fake
def _(value, offsets, weights, reference, shapes, softmax):
    n, queries, m, d = value.shape[0], offsets.shape[1], value.shape[2], value.shape[3]
    return value.new_empty(n, queries, m * d)
