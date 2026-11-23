# Portions adapted from RoMaV2 (MIT, Copyright (c) 2025 Johan Edstedt); see THIRD_PARTY_NOTICES.md.
"""Triton attention kernels: one-launch RoPE for Q/K (ported from RoMaV2 ``rope_fusion.py``) and multi-scale
deformable attention."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


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


# -- multi-scale deformable attention -------------------------------------------------------------------


@triton.jit
def _bilinear_corner(value_ptr, first_token, m, xi, yi, w, h, weight, valid, d, M: tl.constexpr, D: tl.constexpr):
    ok = valid & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    token = first_token + yi.to(tl.int64) * w + xi
    ptrs = value_ptr + ((token * M + m) * D)[:, None] + d[None, :]
    return tl.load(ptrs, mask=ok[:, None], other=0.0).to(tl.float32) * weight[:, None]


@triton.jit
def _deformable_attention_kernel(
    value_ptr,
    offsets_ptr,
    weights_ptr,
    reference_ptr,
    shapes_ptr,
    out_ptr,
    tokens,
    queries,
    offsets_stride_b,
    offsets_stride_q,
    weights_stride_b,
    weights_stride_q,
    M: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    SOFTMAX: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_nm = tl.program_id(1)
    batch = pid_nm // M
    m = pid_nm % M
    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qmask = q < queries
    d = tl.arange(0, D)
    row = batch.to(tl.int64) * queries + q
    offsets_row = offsets_ptr + batch.to(tl.int64) * offsets_stride_b + q.to(tl.int64) * offsets_stride_q + m * (L * P * 2)
    weights_row = weights_ptr + batch.to(tl.int64) * weights_stride_b + q.to(tl.int64) * weights_stride_q + m * (L * P)
    if SOFTMAX:  # softmax over the head's L * P logits (FP32): max pre-pass, normalized once at the end
        row_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        for k in tl.static_range(L * P):
            row_max = tl.maximum(row_max, tl.load(weights_row + k, mask=qmask, other=0.0).to(tl.float32))
        denominator = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    ref_x = tl.load(reference_ptr + row * 2, mask=qmask, other=0.0)
    ref_y = tl.load(reference_ptr + row * 2 + 1, mask=qmask, other=0.0)
    acc = tl.zeros((BLOCK_Q, D), dtype=tl.float32)
    for level in tl.static_range(L):
        h = tl.load(shapes_ptr + level * 3)
        w = tl.load(shapes_ptr + level * 3 + 1)
        start = tl.load(shapes_ptr + level * 3 + 2)
        hf = h.to(tl.float32)
        wf = w.to(tl.float32)
        first_token = batch.to(tl.int64) * tokens + start
        for point in tl.static_range(P):
            k = level * P + point
            off_x = tl.load(offsets_row + k * 2, mask=qmask, other=0.0).to(tl.float32)
            off_y = tl.load(offsets_row + k * 2 + 1, mask=qmask, other=0.0).to(tl.float32)
            attn = tl.load(weights_row + k, mask=qmask, other=0.0).to(tl.float32)
            if SOFTMAX:
                attn = tl.exp(attn - row_max)
                denominator += attn
            # grid_sample(align_corners=False) of a (ref + offset / size) * 2 - 1 grid.
            px = ref_x * wf + off_x - 0.5
            py = ref_y * hf + off_y - 0.5
            x0f = tl.floor(px)
            y0f = tl.floor(py)
            lx = px - x0f
            ly = py - y0f
            x0 = x0f.to(tl.int32)
            y0 = y0f.to(tl.int32)
            acc += _bilinear_corner(value_ptr, first_token, m, x0, y0, w, h, attn * (1 - lx) * (1 - ly), qmask, d, M, D)
            acc += _bilinear_corner(value_ptr, first_token, m, x0 + 1, y0, w, h, attn * lx * (1 - ly), qmask, d, M, D)
            acc += _bilinear_corner(value_ptr, first_token, m, x0, y0 + 1, w, h, attn * (1 - lx) * ly, qmask, d, M, D)
            acc += _bilinear_corner(value_ptr, first_token, m, x0 + 1, y0 + 1, w, h, attn * lx * ly, qmask, d, M, D)
    if SOFTMAX:
        acc = acc / tl.where(qmask, denominator, 1.0)[:, None]
    out_ptrs = out_ptr + ((row * M + m) * D)[:, None] + d[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=qmask[:, None])


@functools.lru_cache(maxsize=64)
def _shape_table(spatial_shapes: tuple[tuple[int, int], ...], device: torch.device) -> Tensor:
    rows, start = [], 0
    for h, w in spatial_shapes:
        rows.append((h, w, start))
        start += h * w
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _query_strides(tensor: Tensor, inner: int) -> tuple[int, int]:
    """(batch, query) strides of a ``[N, Lq, ...]`` view whose trailing ``inner`` elements are contiguous."""
    if tensor[0, 0].reshape(-1).numel() != inner or not tensor[0, 0].is_contiguous():
        raise ValueError("per-query block must be contiguous")
    return tensor.stride(0), tensor.stride(1)


DEFORMABLE_TILE = {"block_q": 32, "num_warps": 4}


def deformable_attention(
    value: Tensor,
    offsets: Tensor,
    weights: Tensor,
    reference: Tensor,
    spatial_shapes: tuple[tuple[int, int], ...],
    softmax: bool = False,
) -> Tensor:
    """See ``flashim.ops.attention._deformable_attention_reference``. FP32 accumulation, output in value dtype.

    ``offsets`` / ``weights`` may be query-strided views (e.g. column slices of one fused projection)."""
    n, tokens, m, d = value.shape
    queries, levels, points = offsets.shape[1], offsets.shape[3], offsets.shape[4]
    value = value.contiguous()
    offsets_strides = _query_strides(offsets, m * levels * points * 2)
    weights_strides = _query_strides(weights, m * levels * points)
    reference = reference.float().contiguous()
    out = torch.empty(n, queries, m * d, dtype=value.dtype, device=value.device)
    block_q = DEFORMABLE_TILE["block_q"]
    grid = (triton.cdiv(queries, block_q), n * m)
    _deformable_attention_kernel[grid](
        value,
        offsets,
        weights,
        reference,
        _shape_table(spatial_shapes, value.device),
        out,
        tokens,
        queries,
        *offsets_strides,
        *weights_strides,
        M=m,
        L=levels,
        P=points,
        D=d,
        SOFTMAX=softmax,
        BLOCK_Q=block_q,
        num_warps=DEFORMABLE_TILE["num_warps"],
    )
    return out

# -- interleaved-pair RoPE with per-batch tables (LightGlue) ---------------------------------------------


@triton.jit
def _rope_pairs_rows(src, s_b, s_h, s_n, dst, b, h, n, d, partner, sign, cos, sin, ok, out_off):
    row = src + b * s_b + h * s_h + n[:, None] * s_n
    value = tl.load(row + d[None, :], mask=ok, other=0.0).to(tl.float32)
    rotated = tl.load(row + partner[None, :], mask=ok, other=0.0).to(tl.float32) * sign[None, :]
    out = libdevice.add_rn(libdevice.mul_rn(value, cos), libdevice.mul_rn(rotated, sin))
    tl.store(dst + out_off, out.to(dst.dtype.element_ty), mask=ok)


@triton.jit
def _rope_qk_interleaved_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
    qs_b, qs_h, qs_n, ks_b, ks_h, ks_n, ts_b, ts_n,
    H, N,
    D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """q and k of one (batch, head, token block) in one program: ``t * cos + rotate_pairs(t) * sin`` in FP32 with
    rounded (non-contracted) products and sum, the operation order of the torch reference, so the result is
    bit-identical to it; the pair partner is the same row's ``d ^ 1`` element (last dim contiguous)."""
    bh = tl.program_id(1)
    b = (bh // H).to(tl.int64)
    h = (bh % H).to(tl.int64)
    n = (tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    d = tl.arange(0, BLOCK_D)  # D even: a pair partner d ^ 1 of a valid d is valid
    partner = d ^ 1
    sign = tl.where((d & 1) == 0, -1.0, 1.0)
    ok = (n < N)[:, None] & (d < D)[None, :]
    t_off = b * ts_b + n[:, None] * ts_n + d[None, :]
    cos = tl.load(cos_ptr + t_off, mask=ok, other=0.0).to(tl.float32)  # the reference promotes them to FP32 too
    sin = tl.load(sin_ptr + t_off, mask=ok, other=0.0).to(tl.float32)
    out_off = ((b * H + h) * N + n[:, None]) * D + d[None, :]
    _rope_pairs_rows(q_ptr, qs_b, qs_h, qs_n, q_out_ptr, b, h, n, d, partner, sign, cos, sin, ok, out_off)
    _rope_pairs_rows(k_ptr, ks_b, ks_h, ks_n, k_out_ptr, b, h, n, d, partner, sign, cos, sin, ok, out_off)


def rope_qk_interleaved(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    batch, heads, tokens, dim = q.shape
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    block_n = 64
    _rope_qk_interleaved_kernel[(triton.cdiv(tokens, block_n), batch * heads)](
        q, k, cos, sin, q_out, k_out,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), cos.stride(0), cos.stride(2),
        heads, tokens, D=dim, BLOCK_D=triton.next_power_of_2(dim), BLOCK_N=block_n, num_warps=4,
    )
    return q_out, k_out


# -- registrations ---------------------------------------------------------------------------------------

_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


def _rope_eligible(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> bool:
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


ops.rope_qk.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_rope_eligible)(rope_qk)


def _rope_interleaved_eligible(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> bool:
    return (
        not torch.is_grad_enabled()
        and q.ndim == 4
        and k.shape == q.shape
        and q.dtype in _FLOATS
        and k.dtype in _FLOATS
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and cos.dtype == sin.dtype
        and cos.dtype in _FLOATS
        and cos.shape == sin.shape == (q.shape[0], 1, q.shape[2], q.shape[3])
        and cos.stride() == sin.stride()
        and cos.stride(-1) == 1
        and q.shape[-1] % 2 == 0
        and q.shape[-1] <= 256
    )


ops.rope_qk_interleaved.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_rope_interleaved_eligible
)(rope_qk_interleaved)


def _deformable_eligible(value, offsets, weights, reference, spatial_shapes, softmax=False) -> bool:
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


@ops.deformable_attention.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_deformable_eligible
)
def _deformable_attention_impl(value, offsets, weights, reference, spatial_shapes, softmax=False) -> Tensor:
    flat = [size for shape in spatial_shapes for size in shape]
    return _deformable_attention_op(value, offsets, weights, reference, flat, softmax)


# Opaque to torch.compile (no tracing into the launcher; strided offset/weight views pass through as-is).
@torch.library.custom_op("flashim::deformable_attention", mutates_args=())
def _deformable_attention_op(
    value: Tensor, offsets: Tensor, weights: Tensor, reference: Tensor, shapes: list[int], softmax: bool
) -> Tensor:
    spatial_shapes = tuple((shapes[i], shapes[i + 1]) for i in range(0, len(shapes), 2))
    return deformable_attention(value, offsets, weights, reference, spatial_shapes, softmax)


@_deformable_attention_op.register_fake
def _(value, offsets, weights, reference, shapes, softmax):
    n, queries, m, d = value.shape[0], offsets.shape[1], value.shape[2], value.shape[3]
    return value.new_empty(n, queries, m * d)


# -- short-window SDPA (L <= 64, D <= 32, high batch*heads; avoids FlashAttention's 128x128 tile padding) -


@triton.jit
def _short_window_sdpa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    qs_b,
    qs_h,
    qs_l,
    ks_b,
    ks_h,
    ks_l,
    vs_b,
    vs_h,
    vs_l,
    os_b,
    os_h,
    os_l,
    H,
    L_Q,
    L_KV,
    scale,
    D: tl.constexpr,
    BLOCK_LQ: tl.constexpr,
    BLOCK_LKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    b = bh // H
    h = bh - b * H
    lq_idx = tl.arange(0, BLOCK_LQ)
    lkv_idx = tl.arange(0, BLOCK_LKV)
    d_idx = tl.arange(0, BLOCK_D)
    lq_ok = lq_idx < L_Q
    lkv_ok = lkv_idx < L_KV
    d_ok = d_idx < D
    q_mask = lq_ok[:, None] & d_ok[None, :]
    kv_mask = lkv_ok[:, None] & d_ok[None, :]

    q = tl.load(q_ptr + b * qs_b + h * qs_h + lq_idx[:, None] * qs_l + d_idx[None, :], mask=q_mask, other=0.0)
    k = tl.load(k_ptr + b * ks_b + h * ks_h + lkv_idx[:, None] * ks_l + d_idx[None, :], mask=kv_mask, other=0.0)
    v = tl.load(v_ptr + b * vs_b + h * vs_h + lkv_idx[:, None] * vs_l + d_idx[None, :], mask=kv_mask, other=0.0)

    scores = tl.dot(q, tl.trans(k)) * scale
    scores = tl.where(lkv_ok[None, :], scores, -float("inf"))
    scores = scores - tl.max(scores, axis=1)[:, None]
    p = tl.exp(scores)
    p = p / tl.sum(p, axis=1)[:, None]
    out = tl.dot(p.to(q.dtype), v)
    tl.store(out_ptr + b * os_b + h * os_h + lq_idx[:, None] * os_l + d_idx[None, :], out.to(q.dtype), mask=q_mask)


def short_window_sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    batch, heads, l_q, head_dim = q.shape
    l_kv = k.shape[2]
    # Output in [B, L_q, H, D].transpose(1, 2) layout so caller's .transpose(1, 2).reshape(B, L_q, H*D) is zero-copy.
    out = torch.empty((batch, l_q, heads, head_dim), device=q.device, dtype=q.dtype).transpose(1, 2)
    block_lq = max(16, triton.next_power_of_2(l_q))
    block_lkv = max(16, triton.next_power_of_2(l_kv))
    block_d = max(16, triton.next_power_of_2(head_dim))
    num_warps = 2 if max(block_lq, block_lkv) <= 32 else 4
    _short_window_sdpa_kernel[(batch * heads,)](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        heads,
        l_q,
        l_kv,
        head_dim**-0.5,
        D=head_dim,
        BLOCK_LQ=block_lq,
        BLOCK_LKV=block_lkv,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out


def _short_window_sdpa_eligible(q: Tensor, k: Tensor, v: Tensor) -> bool:
    return (
        not torch.is_grad_enabled()
        and q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and q.shape[:2] == k.shape[:2] == v.shape[:2]
        and q.shape[3] == k.shape[3] == v.shape[3]
        and k.shape[2] == v.shape[2]
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and 0 < q.shape[2] <= 64
        and 0 < k.shape[2] <= 64
        and 0 < q.shape[3] <= 32
        and q.shape[0] * q.shape[1] >= 256
    )


ops.sdpa.register(
    "triton_short_window", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_short_window_sdpa_eligible
)(short_window_sdpa)


# -- shared-QK bidirectional cross-attention (short windows) ----------------------------------------------


@triton.jit
def _bidirectional_cross_attn_kernel(
    qk0_ptr,
    qk1_ptr,
    v0_ptr,
    v1_ptr,
    out0_ptr,
    out1_ptr,
    s0_m,
    s0_l,
    s1_m,
    s1_l,
    v0_m,
    v0_l,
    v1_m,
    v1_l,
    o0_m,
    o0_l,
    o1_m,
    o1_l,
    H,
    L0,
    L1,
    scale,
    D: tl.constexpr,
    BLOCK_L0: tl.constexpr,
    BLOCK_L1: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    mh = tl.program_id(0).to(tl.int64)
    m = mh // H
    h = mh - m * H
    h_off = h * D
    l0 = tl.arange(0, BLOCK_L0)
    l1 = tl.arange(0, BLOCK_L1)
    d = tl.arange(0, BLOCK_D)
    m0 = l0 < L0
    m1 = l1 < L1
    md = d < D

    qk0 = tl.load(qk0_ptr + m * s0_m + l0[:, None] * s0_l + h_off + d[None, :], mask=m0[:, None] & md[None, :], other=0.0)
    qk1 = tl.load(qk1_ptr + m * s1_m + l1[:, None] * s1_l + h_off + d[None, :], mask=m1[:, None] & md[None, :], other=0.0)
    v0 = tl.load(v0_ptr + m * v0_m + l0[:, None] * v0_l + h_off + d[None, :], mask=m0[:, None] & md[None, :], other=0.0)
    v1 = tl.load(v1_ptr + m * v1_m + l1[:, None] * v1_l + h_off + d[None, :], mask=m1[:, None] & md[None, :], other=0.0)

    sim = tl.dot(qk0, tl.trans(qk1)) * scale

    s01 = tl.where(m1[None, :], sim, -float("inf"))
    s01 = s01 - tl.max(s01, axis=1)[:, None]
    p01 = tl.exp(s01)
    p01 = p01 / tl.sum(p01, axis=1)[:, None]
    out0 = tl.dot(p01.to(v1.dtype), v1)
    tl.store(
        out0_ptr + m * o0_m + l0[:, None] * o0_l + h_off + d[None, :],
        out0.to(v1.dtype),
        mask=m0[:, None] & md[None, :],
    )

    s10 = tl.where(m0[:, None], sim, -float("inf"))
    s10 = s10 - tl.max(s10, axis=0)[None, :]
    p10 = tl.exp(s10)
    p10 = p10 / tl.sum(p10, axis=0)[None, :]
    out1 = tl.dot(tl.trans(p10).to(v0.dtype), v0)
    tl.store(
        out1_ptr + m * o1_m + l1[:, None] * o1_l + h_off + d[None, :],
        out1.to(v0.dtype),
        mask=m1[:, None] & md[None, :],
    )


def bidirectional_cross_attention(
    qk0: Tensor, qk1: Tensor, v0: Tensor, v1: Tensor, num_heads: int
) -> tuple[Tensor, Tensor]:
    m, l0, c = qk0.shape
    l1 = qk1.shape[1]
    d = c // num_heads
    if m == 0:
        return torch.empty_like(v0), torch.empty_like(v1)
    out0 = torch.empty((m, l0, c), device=v0.device, dtype=v0.dtype)
    out1 = torch.empty((m, l1, c), device=v1.device, dtype=v1.dtype)
    block_l0 = max(16, triton.next_power_of_2(l0))
    block_l1 = max(16, triton.next_power_of_2(l1))
    block_d = max(16, triton.next_power_of_2(d))
    num_warps = 2 if max(block_l0, block_l1) <= 32 else 4
    _bidirectional_cross_attn_kernel[(m * num_heads,)](
        qk0,
        qk1,
        v0,
        v1,
        out0,
        out1,
        qk0.stride(0),
        qk0.stride(1),
        qk1.stride(0),
        qk1.stride(1),
        v0.stride(0),
        v0.stride(1),
        v1.stride(0),
        v1.stride(1),
        out0.stride(0),
        out0.stride(1),
        out1.stride(0),
        out1.stride(1),
        num_heads,
        l0,
        l1,
        d**-0.5,
        D=d,
        BLOCK_L0=block_l0,
        BLOCK_L1=block_l1,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out0, out1


def _bidirectional_cross_attn_eligible(qk0: Tensor, qk1: Tensor, v0: Tensor, v1: Tensor, num_heads: int) -> bool:
    if torch.is_grad_enabled() or qk0.ndim != 3 or qk1.ndim != 3 or v0.shape != qk0.shape or v1.shape != qk1.shape:
        return False
    if qk0.dtype not in (torch.bfloat16, torch.float16) or not (qk0.dtype == qk1.dtype == v0.dtype == v1.dtype):
        return False
    if not all(t.stride(-1) == 1 for t in (qk0, qk1, v0, v1)):
        return False
    d = qk0.shape[2] // num_heads
    return qk0.shape[2] % num_heads == 0 and 0 < qk0.shape[1] <= 64 and 0 < qk1.shape[1] <= 64 and 0 < d <= 32


ops.bidirectional_cross_attention.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_bidirectional_cross_attn_eligible
)(bidirectional_cross_attention)


# -- ELU+1 linear attention ------------------------------------------------------------------------------


@triton.jit
def _linear_attn_small_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    qs_n, qs_l, qs_h, ks_n, ks_s, ks_h, vs_n, vs_s, vs_h,
    H, L, S, eps,
    D: tl.constexpr, V: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
    BF16_DOT: tl.constexpr,
):
    """Single-pass ELU+1 linear attention for short sequences (``L, S <= 64``, e.g. LoFTR fine 5x5 windows)."""
    nh = tl.program_id(0)
    n = (nh // H).to(tl.int64)
    h = (nh % H).to(tl.int64)
    l_idx = tl.arange(0, BLOCK_L)
    s_idx = tl.arange(0, BLOCK_S)
    d_idx = tl.arange(0, BLOCK_D)
    v_idx = tl.arange(0, BLOCK_V)
    l_ok = l_idx < L
    s_ok = s_idx < S
    d_ok = d_idx < D
    v_ok = v_idx < V

    k_ptrs = k_ptr + n * ks_n + h * ks_h + s_idx[:, None].to(tl.int64) * ks_s + d_idx[None, :]
    v_ptrs = v_ptr + n * vs_n + h * vs_h + s_idx[:, None].to(tl.int64) * vs_s + v_idx[None, :]
    k = tl.load(k_ptrs, mask=s_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=s_ok[:, None] & v_ok[None, :], other=0.0).to(tl.float32)
    # elu(x) + 1 = x + 1 for x > 0 else exp(x); zero out padded positions after the feature map.
    kf = tl.where(s_ok[:, None] & d_ok[None, :], tl.where(k > 0.0, k + 1.0, tl.exp(k)), 0.0)
    k_sum = tl.sum(kf, axis=0)
    inv_s = 1.0 / S
    v_scaled = v * inv_s
    if BF16_DOT:
        kv = tl.dot(tl.trans(kf.to(tl.bfloat16)), v_scaled.to(tl.bfloat16))
    else:
        kv = tl.dot(tl.trans(kf), v_scaled, input_precision="ieee")

    q_ptrs = q_ptr + n * qs_n + h * qs_h + l_idx[:, None].to(tl.int64) * qs_l + d_idx[None, :]
    q = tl.load(q_ptrs, mask=l_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
    qf = tl.where(l_ok[:, None] & d_ok[None, :], tl.where(q > 0.0, q + 1.0, tl.exp(q)), 0.0)
    denom = tl.sum(qf * k_sum[None, :], axis=1) + eps
    if BF16_DOT:
        acc = tl.dot(qf.to(tl.bfloat16), kv.to(tl.bfloat16))
    else:
        acc = tl.dot(qf, kv, input_precision="ieee")
    out = acc * (S / denom)[:, None]
    out_ptrs = out_ptr + ((n * L + l_idx[:, None].to(tl.int64)) * H + h) * V + v_idx[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=l_ok[:, None] & v_ok[None, :])


@triton.jit
def _linear_attn_kv_split_kernel(
    k_ptr, v_ptr, partial_kv_ptr, partial_ksum_ptr,
    ks_n, ks_s, ks_h, vs_n, vs_s, vs_h,
    H, S, NUM_SPLITS,
    D: tl.constexpr, V: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
    BF16_DOT: tl.constexpr,
):
    """Split-S accumulation of ``k_sum = sum_s (elu(k) + 1)`` and ``kv = sum_s (elu(k) + 1)^T (v / S)``."""
    split = tl.program_id(0)
    nh = tl.program_id(1)
    n = (nh // H).to(tl.int64)
    h = (nh % H).to(tl.int64)
    d_idx = tl.arange(0, BLOCK_D)
    v_idx = tl.arange(0, BLOCK_V)
    d_ok = d_idx < D
    v_ok = v_idx < V
    inv_s = 1.0 / S

    k_sum = tl.zeros((BLOCK_D,), dtype=tl.float32)
    kv = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    k_base = k_ptr + n * ks_n + h * ks_h
    v_base = v_ptr + n * vs_n + h * vs_h

    step = NUM_SPLITS * BLOCK_S
    for s0 in range(split * BLOCK_S, S, step):
        s_idx = s0 + tl.arange(0, BLOCK_S)
        s_ok = s_idx < S
        k = tl.load(k_base + s_idx[:, None].to(tl.int64) * ks_s + d_idx[None, :], mask=s_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_base + s_idx[:, None].to(tl.int64) * vs_s + v_idx[None, :], mask=s_ok[:, None] & v_ok[None, :], other=0.0).to(tl.float32)
        kf = tl.where(s_ok[:, None] & d_ok[None, :], tl.where(k > 0.0, k + 1.0, tl.exp(k)), 0.0)
        k_sum += tl.sum(kf, axis=0)
        v_scaled = v * inv_s
        if BF16_DOT:
            kv += tl.dot(tl.trans(kf.to(tl.bfloat16)), v_scaled.to(tl.bfloat16))
        else:
            kv += tl.dot(tl.trans(kf), v_scaled, input_precision="ieee")

    slot = nh.to(tl.int64) * NUM_SPLITS + split
    tl.store(partial_ksum_ptr + slot * BLOCK_D + d_idx, k_sum, mask=d_ok)
    tl.store(
        partial_kv_ptr + slot * (BLOCK_D * BLOCK_V) + d_idx[:, None] * BLOCK_V + v_idx[None, :],
        kv,
        mask=d_ok[:, None] & v_ok[None, :],
    )


@triton.jit
def _linear_attn_reduce_kernel(
    partial_kv_ptr, partial_ksum_ptr, kv_ptr, ksum_ptr,
    NUM_SPLITS, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    nh = tl.program_id(0).to(tl.int64)
    d_idx = tl.arange(0, BLOCK_D)
    v_idx = tl.arange(0, BLOCK_V)
    k_sum = tl.zeros((BLOCK_D,), dtype=tl.float32)
    kv = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    for split in range(NUM_SPLITS):
        slot = nh * NUM_SPLITS + split
        k_sum += tl.load(partial_ksum_ptr + slot * BLOCK_D + d_idx)
        kv += tl.load(partial_kv_ptr + slot * (BLOCK_D * BLOCK_V) + d_idx[:, None] * BLOCK_V + v_idx[None, :])
    tl.store(ksum_ptr + nh * BLOCK_D + d_idx, k_sum)
    tl.store(kv_ptr + nh * (BLOCK_D * BLOCK_V) + d_idx[:, None] * BLOCK_V + v_idx[None, :], kv)


@triton.jit
def _linear_attn_q_kernel(
    q_ptr, kv_ptr, ksum_ptr, out_ptr,
    qs_n, qs_l, qs_h,
    H, L, S, eps,
    D: tl.constexpr, V: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
    BF16_DOT: tl.constexpr,
):
    """``((elu(q) + 1) @ kv) * (S / ((elu(q) + 1) . k_sum + eps))`` in one fused pass over ``q``."""
    pid_l = tl.program_id(0)
    nh = tl.program_id(1)
    n = (nh // H).to(tl.int64)
    h = (nh % H).to(tl.int64)
    l_idx = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    d_idx = tl.arange(0, BLOCK_D)
    v_idx = tl.arange(0, BLOCK_V)
    l_ok = l_idx < L
    d_ok = d_idx < D
    v_ok = v_idx < V

    k_sum = tl.load(ksum_ptr + nh.to(tl.int64) * BLOCK_D + d_idx, mask=d_ok, other=0.0)
    kv = tl.load(
        kv_ptr + nh.to(tl.int64) * (BLOCK_D * BLOCK_V) + d_idx[:, None] * BLOCK_V + v_idx[None, :],
        mask=d_ok[:, None] & v_ok[None, :],
        other=0.0,
    )
    q_ptrs = q_ptr + n * qs_n + h * qs_h + l_idx[:, None].to(tl.int64) * qs_l + d_idx[None, :]
    q = tl.load(q_ptrs, mask=l_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
    qf = tl.where(l_ok[:, None] & d_ok[None, :], tl.where(q > 0.0, q + 1.0, tl.exp(q)), 0.0)
    denom = tl.sum(qf * k_sum[None, :], axis=1) + eps
    if BF16_DOT:
        acc = tl.dot(qf.to(tl.bfloat16), kv.to(tl.bfloat16))
    else:
        acc = tl.dot(qf, kv, input_precision="ieee")
    out = acc * (S / denom)[:, None]
    out_ptrs = out_ptr + ((n * L + l_idx[:, None].to(tl.int64)) * H + h) * V + v_idx[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=l_ok[:, None] & v_ok[None, :])


def elu_linear_attention(
    q: Tensor, k: Tensor, v: Tensor, eps: float = 1e-6, gemm_dtype: torch.dtype | None = None
) -> Tensor:
    """Fused ELU+1 linear attention ``[N, L, H, D]`` x ``[N, S, H, D]`` x ``[N, S, H, V]`` -> ``[N, L, H, V]``."""
    n, seq_l, heads, d = q.shape
    seq_s, v_dim = k.shape[1], v.shape[3]
    device = q.device
    out = torch.empty((n, seq_l, heads, v_dim), device=device, dtype=q.dtype)
    if seq_l == 0 or seq_s == 0:
        return out.zero_()

    block_d = max(16, triton.next_power_of_2(d))
    block_v = max(16, triton.next_power_of_2(v_dim))
    bf16_dot = gemm_dtype == torch.bfloat16

    if seq_l <= 64 and seq_s <= 64:
        block_l = max(16, triton.next_power_of_2(seq_l))
        block_s = max(16, triton.next_power_of_2(seq_s))
        _linear_attn_small_kernel[(n * heads,)](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            heads, seq_l, seq_s, float(eps),
            D=d, V=v_dim, BLOCK_L=block_l, BLOCK_S=block_s, BLOCK_D=block_d, BLOCK_V=block_v,
            BF16_DOT=bf16_dot, num_warps=4,
        )
        return out

    # Swept over BLOCK_S/BLOCK_L in {64, 128, 256}, splits in {8, 16, 32}, warps in {4, 8} on sm_120
    # for (1, 8190, 8, 32) BF16: (64, 128, 32, 4) measured 31.4 us (vs 196 us BF16 torch, 382 us FP32).
    block_s = 64
    block_l = 64 if (not bf16_dot and max(block_d, block_v) > 32) else 128
    max_splits = 32 if n * heads <= 16 else 8
    num_splits = max(1, min(triton.cdiv(seq_s, block_s), max_splits))

    if num_splits == 1:
        kv = torch.empty((n * heads, block_d, block_v), device=device, dtype=torch.float32)
        k_sum = torch.empty((n * heads, block_d), device=device, dtype=torch.float32)
        _linear_attn_kv_split_kernel[(1, n * heads)](
            k, v, kv, k_sum,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            heads, seq_s, 1,
            D=d, V=v_dim, BLOCK_S=block_s, BLOCK_D=block_d, BLOCK_V=block_v,
            BF16_DOT=bf16_dot, num_warps=4,
        )
    else:
        partial_kv = torch.empty((n * heads, num_splits, block_d, block_v), device=device, dtype=torch.float32)
        partial_ksum = torch.empty((n * heads, num_splits, block_d), device=device, dtype=torch.float32)
        _linear_attn_kv_split_kernel[(num_splits, n * heads)](
            k, v, partial_kv, partial_ksum,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            heads, seq_s, num_splits,
            D=d, V=v_dim, BLOCK_S=block_s, BLOCK_D=block_d, BLOCK_V=block_v,
            BF16_DOT=bf16_dot, num_warps=4,
        )
        kv = torch.empty((n * heads, block_d, block_v), device=device, dtype=torch.float32)
        k_sum = torch.empty((n * heads, block_d), device=device, dtype=torch.float32)
        _linear_attn_reduce_kernel[(n * heads,)](
            partial_kv, partial_ksum, kv, k_sum,
            num_splits, BLOCK_D=block_d, BLOCK_V=block_v, num_warps=4,
        )

    _linear_attn_q_kernel[(triton.cdiv(seq_l, block_l), n * heads)](
        q, kv, k_sum, out,
        q.stride(0), q.stride(1), q.stride(2),
        heads, seq_l, seq_s, float(eps),
        D=d, V=v_dim, BLOCK_L=block_l, BLOCK_D=block_d, BLOCK_V=block_v,
        BF16_DOT=bf16_dot, num_warps=4,
    )
    return out


def _elu_linear_attn_eligible(
    q: Tensor, k: Tensor, v: Tensor, eps: float = 1e-6, gemm_dtype: torch.dtype | None = None
) -> bool:
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return False  # forward-only kernel
    return (
        q.ndim == k.ndim == v.ndim == 4
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[2] == k.shape[2] == v.shape[2]
        and k.shape[1] == v.shape[1]
        and q.shape[3] == k.shape[3]
        and q.dtype in _FLOATS
        and k.dtype in _FLOATS
        and v.dtype in _FLOATS
        and gemm_dtype in (None, torch.bfloat16)
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and 0 < q.shape[3] <= 64
        and 0 < v.shape[3] <= 64
    )


ops.elu_linear_attention.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_elu_linear_attn_eligible
)(elu_linear_attention)
