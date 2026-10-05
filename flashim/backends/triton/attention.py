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
