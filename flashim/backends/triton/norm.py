"""Triton normalization kernels: fused InstanceNorm2d + PReLU (DISK), RGB-mean + InstanceNorm2d (+ unfold, XFeat),
and multi-row LayerNorm fused with residual add, left-concatenation, or window positional encoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature

# -- InstanceNorm2d + PReLU (NCHW input path) ---------------------------------------------------------


@triton.jit
def _in_stats_nchw_kernel(
    x_ptr, cnt_ptr, mean_ptr, m2_ptr, S, G,
    BLOCK: tl.constexpr,
):
    """Pass 1 (NCHW): program ``(g, bc)`` computes exact Welford ``(count, mean, m2)`` over chunk ``g`` of plane ``bc``."""
    g = tl.program_id(0)
    bc = tl.program_id(1)
    chunk = tl.cdiv(S, G)
    start = g * chunk
    end = tl.minimum(start + chunk, S)
    base = x_ptr + bc.to(tl.int64) * S

    count = 0.0
    mean = 0.0
    m2 = 0.0
    for off in range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        val = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
        bn = tl.sum(mask.to(tl.float32), axis=0)
        bm = tl.sum(val, axis=0) / tl.maximum(bn, 1.0)
        diff = tl.where(mask, val - bm, 0.0)
        bm2 = tl.sum(diff * diff, axis=0)
        new_count = count + bn
        delta = bm - mean
        mean = tl.where(new_count > 0.0, mean + delta * (bn / tl.maximum(new_count, 1.0)), 0.0)
        m2 = m2 + bm2 + delta * delta * (count * bn / tl.maximum(new_count, 1.0))
        count = new_count

    slot = bc * G + g
    tl.store(cnt_ptr + slot, count)
    tl.store(mean_ptr + slot, mean)
    tl.store(m2_ptr + slot, m2)


@triton.jit
def _in_prelu_nchw_kernel(
    x_ptr, w_ptr, cnt_ptr, mean_ptr, m2_ptr, out_ptr, C, S, eps,
    G: tl.constexpr, BLOCK: tl.constexpr, OUT_NHWC: tl.constexpr,
):
    """Pass 2 (NCHW input): merges ``G`` partials in registers, normalizes + PReLU, writes NCHW or NHWC."""
    pid_s = tl.program_id(0)
    bc = tl.program_id(1)
    b = bc // C
    c = bc % C

    gs = tl.arange(0, G)
    base_slot = bc * G + gs
    cnt = tl.load(cnt_ptr + base_slot)
    m = tl.load(mean_ptr + base_slot)
    m2 = tl.load(m2_ptr + base_slot)
    total = tl.sum(cnt, axis=0)
    mean = tl.sum(cnt * m, axis=0) / tl.maximum(total, 1.0)
    dm = m - mean
    var = tl.sum(m2 + cnt * dm * dm, axis=0) / tl.maximum(total, 1.0)
    # Inductor passes Python float scalars as fp64; cast to fp32 so sqrt_rn/div_rn stay fp32 under torch.compile.
    inv_std = tl.math.div_rn(1.0, tl.math.sqrt_rn(var + tl.cast(eps, tl.float32)))
    w = tl.load(w_ptr + c).to(tl.float32)

    idx = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = idx < S
    val = tl.load(x_ptr + bc.to(tl.int64) * S + idx, mask=mask, other=0.0).to(tl.float32)
    # Match unfused nn.InstanceNorm2d -> nn.PReLU intermediate rounding (no-op for FP32).
    y = ((val - mean) * inv_std).to(out_ptr.dtype.element_ty).to(tl.float32)
    y = tl.where(y > 0.0, y, y * w)
    if OUT_NHWC:
        out_off = (b.to(tl.int64) * S + idx.to(tl.int64)) * C + c
    else:
        out_off = bc.to(tl.int64) * S + idx
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=mask)


# -- InstanceNorm2d + PReLU (NHWC input path) ---------------------------------------------------------


@triton.jit
def _in_stats_nhwc_kernel(
    x_ptr, cnt_ptr, mean_ptr, m2_ptr, C, S, G,
    BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Pass 1 (NHWC): program ``(g, b)`` reduces all ``C`` channels over pixel chunk ``g`` with coalesced channel loads."""
    g = tl.program_id(0)
    b = tl.program_id(1)
    chunk = tl.cdiv(S, G)
    start = g * chunk
    end = tl.minimum(start + chunk, S)
    base = x_ptr + b.to(tl.int64) * S * C

    cs = tl.arange(0, BLOCK_C)
    c_mask = cs < C
    count = 0.0
    mean = tl.zeros((BLOCK_C,), dtype=tl.float32)
    m2 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for off in range(start, end, BLOCK_P):
        ps = off + tl.arange(0, BLOCK_P)
        p_mask = ps < end
        bn = tl.sum(p_mask.to(tl.float32), axis=0)
        val = tl.load(base + ps[:, None].to(tl.int64) * C + cs[None, :],
                      mask=p_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
        bm = tl.sum(val, axis=0) / tl.maximum(bn, 1.0)
        diff = tl.where(p_mask[:, None], val - bm[None, :], 0.0)
        bm2 = tl.sum(diff * diff, axis=0)
        new_count = count + bn
        delta = bm - mean
        mean = tl.where(new_count > 0.0, mean + delta * (bn / tl.maximum(new_count, 1.0)), 0.0)
        m2 = m2 + bm2 + delta * delta * (count * bn / tl.maximum(new_count, 1.0))
        count = new_count

    tl.store(cnt_ptr + b * G + g, count)
    slot = (b * G + g) * C + cs
    tl.store(mean_ptr + slot, mean, mask=c_mask)
    tl.store(m2_ptr + slot, m2, mask=c_mask)


@triton.jit
def _in_prelu_nhwc_kernel(
    x_ptr, w_ptr, cnt_ptr, mean_ptr, m2_ptr, out_ptr, C, S, eps,
    G: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr, OUT_NHWC: tl.constexpr,
):
    """Pass 2 (NHWC input): merges ``G`` partials per channel, normalizes + PReLU, writes NHWC or NCHW (both coalesced!)."""
    pid_p = tl.program_id(0)
    b = tl.program_id(1)
    cs = tl.arange(0, BLOCK_C)
    c_mask = cs < C
    gs = tl.arange(0, G)

    cnt = tl.load(cnt_ptr + b * G + gs)  # [G]
    total = tl.sum(cnt, axis=0)
    slots = (b * G + gs[:, None]) * C + cs[None, :]
    m = tl.load(mean_ptr + slots, mask=c_mask[None, :], other=0.0)  # [G, BLOCK_C]
    m2 = tl.load(m2_ptr + slots, mask=c_mask[None, :], other=0.0)
    mean = tl.sum(cnt[:, None] * m, axis=0) / tl.maximum(total, 1.0)  # [BLOCK_C]
    dm = m - mean[None, :]
    var = tl.sum(m2 + cnt[:, None] * dm * dm, axis=0) / tl.maximum(total, 1.0)
    inv_std = tl.math.div_rn(1.0, tl.math.sqrt_rn(var + tl.cast(eps, tl.float32)))
    w = tl.load(w_ptr + cs, mask=c_mask, other=0.0).to(tl.float32)

    ps = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = ps < S
    ok = p_mask[:, None] & c_mask[None, :]
    val = tl.load(x_ptr + (b.to(tl.int64) * S + ps[:, None].to(tl.int64)) * C + cs[None, :],
                  mask=ok, other=0.0).to(tl.float32)
    # Match unfused nn.InstanceNorm2d -> nn.PReLU intermediate rounding (no-op for FP32).
    y = ((val - mean[None, :]) * inv_std[None, :]).to(out_ptr.dtype.element_ty).to(tl.float32)
    y = tl.where(y > 0.0, y, y * w[None, :])
    if OUT_NHWC:
        out_off = (b.to(tl.int64) * S + ps[:, None].to(tl.int64)) * C + cs[None, :]
    else:
        out_off = (b.to(tl.int64) * C + cs[None, :].to(tl.int64)) * S + ps[:, None].to(tl.int64)
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=ok)


def _pick_chunks(spatial: int, max_chunks: int = 32) -> int:
    g = 1
    while g < max_chunks and spatial >= g * 8192:
        g *= 2
    return g


def _instance_norm_prelu_triton(
    x: Tensor, weight: Tensor, eps: float = 1e-5, *, out_channels_last: bool | None = None
) -> Tensor:
    b, c, h, w = x.shape
    s = h * w
    is_nhwc = x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous()
    out_nhwc = is_nhwc if out_channels_last is None else out_channels_last
    fmt = torch.channels_last if out_nhwc else torch.contiguous_format
    out = torch.empty(b, c, h, w, device=x.device, dtype=x.dtype, memory_format=fmt)
    w_contig = weight.contiguous()

    if is_nhwc and c <= 256:
        g = _pick_chunks(s, max_chunks=64)
        block_c = triton.next_power_of_2(c)
        cnt = torch.empty(b, g, device=x.device, dtype=torch.float32)
        mean = torch.empty(b, g, c, device=x.device, dtype=torch.float32)
        m2 = torch.empty(b, g, c, device=x.device, dtype=torch.float32)
        block_p_stats = 64 if block_c <= 64 else 32
        _in_stats_nhwc_kernel[(g, b)](
            x, cnt, mean, m2, c, s, g, BLOCK_P=block_p_stats, BLOCK_C=block_c, num_warps=4,
        )
        block_p_out = 32 if not out_nhwc else (64 if block_c <= 64 else 32)
        _in_prelu_nhwc_kernel[(triton.cdiv(s, block_p_out), b)](
            x, w_contig, cnt, mean, m2, out, c, s, eps,
            G=g, BLOCK_P=block_p_out, BLOCK_C=block_c, OUT_NHWC=out_nhwc, num_warps=4,
        )
        return out

    x_nchw = x.contiguous()
    g = _pick_chunks(s, max_chunks=32)
    bc = b * c
    cnt = torch.empty(bc, g, device=x.device, dtype=torch.float32)
    mean = torch.empty(bc, g, device=x.device, dtype=torch.float32)
    m2 = torch.empty(bc, g, device=x.device, dtype=torch.float32)
    _in_stats_nchw_kernel[(g, bc)](x_nchw, cnt, mean, m2, s, g, BLOCK=1024, num_warps=4)
    block = 1024
    _in_prelu_nchw_kernel[(triton.cdiv(s, block), bc)](
        x_nchw, w_contig, cnt, mean, m2, out, c, s, eps,
        G=g, BLOCK=block, OUT_NHWC=out_nhwc, num_warps=4,
    )
    return out


def _in_prelu_eligible(x: Tensor, weight: Tensor, eps: float = 1e-5, *, out_channels_last: bool | None = None) -> bool:
    return (
        x.is_cuda and x.ndim == 4 and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and weight.ndim == 1 and weight.shape[0] == x.shape[1] and x.shape[2] * x.shape[3] > 0
    )


ops.instance_norm_prelu.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_in_prelu_eligible
)(_instance_norm_prelu_triton)


# -- RGB-mean + 1-ch InstanceNorm2d (+ optional pixel_unshuffle) for XFeat ----------------------------


@triton.jit
def _rgb_in_stats_kernel(
    img_ptr, cnt_ptr, mean_ptr, m2_ptr, S, G, s_b, s_c,
    BLOCK: tl.constexpr,
):
    """Pass 1: reads contiguous NCHW ``[B, 3, H, W]``, averages RGB on the fly, accumulates chunk Welford stats."""
    g = tl.program_id(0)
    b = tl.program_id(1)
    chunk = tl.cdiv(S, G)
    start = g * chunk
    end = tl.minimum(start + chunk, S)
    base = img_ptr + b.to(tl.int64) * s_b
    inv3 = 1.0 / 3.0

    count = 0.0
    mean = 0.0
    m2 = 0.0
    for off in range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        r = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
        gv = tl.load(base + s_c + idx, mask=mask, other=0.0).to(tl.float32)
        bv = tl.load(base + 2 * s_c + idx, mask=mask, other=0.0).to(tl.float32)
        val = (r + gv + bv) * inv3
        bn = tl.sum(mask.to(tl.float32), axis=0)
        bm = tl.sum(val, axis=0) / tl.maximum(bn, 1.0)
        diff = tl.where(mask, val - bm, 0.0)
        bm2 = tl.sum(diff * diff, axis=0)
        new_count = count + bn
        delta = bm - mean
        mean = tl.where(new_count > 0.0, mean + delta * (bn / tl.maximum(new_count, 1.0)), 0.0)
        m2 = m2 + bm2 + delta * delta * (count * bn / tl.maximum(new_count, 1.0))
        count = new_count

    slot = b * G + g
    tl.store(cnt_ptr + slot, count)
    tl.store(mean_ptr + slot, mean)
    tl.store(m2_ptr + slot, m2)


@triton.jit
def _rgb_in_apply_kernel(
    img_ptr, cnt_ptr, mean_ptr, m2_ptr, gray_ptr, unfold_ptr,
    H, W, eps, s_b, s_c,
    G: tl.constexpr, BLOCK: tl.constexpr, UNFOLD: tl.constexpr,
):
    """Pass 2: merges ``G`` stats, normalizes RGB mean, writes ``gray`` ``[B, 1, H, W]`` and (when ``UNFOLD > 0``)
    ``pixel_unshuffle(gray, UNFOLD)`` ``[B, UNFOLD^2, H//UNFOLD, W//UNFOLD]`` in one pass."""
    pid_s = tl.program_id(0)
    b = tl.program_id(1)
    S = H * W

    gs = tl.arange(0, G)
    slot = b * G + gs
    cnt = tl.load(cnt_ptr + slot)
    m = tl.load(mean_ptr + slot)
    m2 = tl.load(m2_ptr + slot)
    total = tl.sum(cnt, axis=0)
    mean = tl.sum(cnt * m, axis=0) / tl.maximum(total, 1.0)
    dm = m - mean
    var = tl.sum(m2 + cnt * dm * dm, axis=0) / tl.maximum(total, 1.0)
    # Inductor passes Python float scalars as fp64; cast to fp32 so sqrt_rn/div_rn stay fp32 under torch.compile.
    inv_std = tl.math.div_rn(1.0, tl.math.sqrt_rn(var + tl.cast(eps, tl.float32)))
    inv3 = 1.0 / 3.0

    idx = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = idx < S
    base = img_ptr + b.to(tl.int64) * s_b
    r = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(base + s_c + idx, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(base + 2 * s_c + idx, mask=mask, other=0.0).to(tl.float32)
    y = ((r + gv + bv) * inv3 - mean) * inv_std
    y_out = y.to(gray_ptr.dtype.element_ty)
    tl.store(gray_ptr + b.to(tl.int64) * S + idx, y_out, mask=mask)

    if UNFOLD > 0:
        row = idx // W
        col = idx % W
        uh = H // UNFOLD
        uw = W // UNFOLD
        ch = (row % UNFOLD) * UNFOLD + (col % UNFOLD)
        uy = row // UNFOLD
        ux = col // UNFOLD
        u_off = ((b.to(tl.int64) * (UNFOLD * UNFOLD) + ch) * uh + uy) * uw + ux
        tl.store(unfold_ptr + u_off, y_out, mask=mask)


def _rgb_mean_instance_norm_triton(
    image: Tensor, eps: float = 1e-5, *, dtype: torch.dtype | None = None, unfold: int = 0
) -> Tensor | tuple[Tensor, Tensor]:
    img = image.contiguous()
    b, _, h, w = img.shape
    s = h * w
    out_dtype = img.dtype if dtype is None else dtype
    g = _pick_chunks(s, max_chunks=64)
    cnt = torch.empty(b, g, device=img.device, dtype=torch.float32)
    mean = torch.empty(b, g, device=img.device, dtype=torch.float32)
    m2 = torch.empty(b, g, device=img.device, dtype=torch.float32)
    _rgb_in_stats_kernel[(g, b)](img, cnt, mean, m2, s, g, img.stride(0), img.stride(1), BLOCK=1024, num_warps=4)
    gray = torch.empty(b, 1, h, w, device=img.device, dtype=out_dtype)
    unfolded = (
        torch.empty(b, unfold * unfold, h // unfold, w // unfold, device=img.device, dtype=out_dtype)
        if unfold > 0 else gray
    )
    block = 1024
    _rgb_in_apply_kernel[(triton.cdiv(s, block), b)](
        img, cnt, mean, m2, gray, unfolded, h, w, eps, img.stride(0), img.stride(1),
        G=g, BLOCK=block, UNFOLD=unfold, num_warps=4,
    )
    return (gray, unfolded) if unfold > 0 else gray


def _rgb_in_eligible(image: Tensor, eps: float = 1e-5, *, dtype: torch.dtype | None = None, unfold: int = 0) -> bool:
    if not (image.is_cuda and image.ndim == 4 and image.shape[1] == 3 and image.dtype in (torch.float32, torch.bfloat16, torch.float16)):
        return False
    if unfold > 0 and (image.shape[2] % unfold != 0 or image.shape[3] % unfold != 0):
        return False
    return image.shape[2] * image.shape[3] > 0


ops.rgb_mean_instance_norm.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_rgb_in_eligible
)(_rgb_mean_instance_norm_triton)


@triton.jit
def _residual_layernorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    rows,
    eps,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    row_ok = row < rows
    col_ok = col < C
    mask = row_ok[:, None] & col_ok[None, :]
    offs = row[:, None] * C + col[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    diff = tl.where(col_ok[None, :], x - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / C
    rstd = tl.rsqrt(var + tl.cast(eps, tl.float32))
    w = tl.load(w_ptr + col, mask=col_ok, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=col_ok, other=0.0).to(tl.float32)
    y = (diff * rstd[:, None]) * w[None, :] + b[None, :]
    y = y.to(out_ptr.dtype.element_ty)
    if HAS_RESIDUAL:
        res = tl.load(res_ptr + offs, mask=mask, other=0.0)
        y = (res.to(tl.float32) + y.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _cat_layernorm_kernel(
    left_ptr,
    right_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    rows,
    eps,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    row_ok = row < rows
    col_ok = col < C
    mask = row_ok[:, None] & col_ok[None, :]
    in_offs = row[:, None] * C + col[None, :]
    out_base = row[:, None] * (2 * C) + col[None, :]

    left = tl.load(left_ptr + in_offs, mask=mask, other=0.0)
    tl.store(out_ptr + out_base, left, mask=mask)

    r = tl.load(right_ptr + in_offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(r, axis=1) / C
    diff = tl.where(col_ok[None, :], r - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / C
    rstd = tl.rsqrt(var + tl.cast(eps, tl.float32))
    w = tl.load(w_ptr + col, mask=col_ok, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=col_ok, other=0.0).to(tl.float32)
    y = ((diff * rstd[:, None]) * w[None, :] + b[None, :]).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_base + C, y, mask=mask)


def residual_layernorm(
    x: Tensor, residual: Tensor | None, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> Tensor:
    c = x.shape[-1]
    rows = x.numel() // c
    if rows == 0:
        return torch.empty_like(x)
    x_c = x.contiguous()
    res_c = x_c if residual is None else residual.contiguous()
    out = torch.empty_like(x_c)
    block_n = triton.next_power_of_2(c)
    block_m = 8 if block_n <= 256 else 4
    _residual_layernorm_kernel[(triton.cdiv(rows, block_m),)](
        x_c,
        res_c,
        weight,
        bias,
        out,
        rows,
        eps,
        C=c,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HAS_RESIDUAL=residual is not None,
        num_warps=4,
    )
    return out


def cat_layernorm(left: Tensor, right: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    c = right.shape[-1]
    rows = right.numel() // c
    out_shape = (*right.shape[:-1], 2 * c)
    if rows == 0:
        return right.new_empty(out_shape)
    left_c = left.contiguous()
    right_c = right.contiguous()
    out = right.new_empty(out_shape)
    block_n = triton.next_power_of_2(c)
    block_m = 8 if block_n <= 256 else 4
    _cat_layernorm_kernel[(triton.cdiv(rows, block_m),)](
        left_c,
        right_c,
        weight,
        bias,
        out,
        rows,
        eps,
        C=c,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


def _residual_ln_eligible(
    x: Tensor, residual: Tensor | None, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> bool:
    if torch.is_grad_enabled() or x.dtype not in _FLOATS or x.shape[-1] > 1024:
        return False
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        return False
    return residual is None or (residual.shape == x.shape and residual.dtype == x.dtype)


def _cat_ln_eligible(left: Tensor, right: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> bool:
    return (
        not torch.is_grad_enabled()
        and right.dtype in _FLOATS
        and left.shape == right.shape
        and left.dtype == right.dtype
        and weight.dtype == right.dtype
        and bias.dtype == right.dtype
        and right.shape[-1] <= 1024
    )


ops.residual_layernorm.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_residual_ln_eligible
)(residual_layernorm)
ops.cat_layernorm.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_cat_ln_eligible
)(cat_layernorm)


@triton.jit
def _pos_cat_layernorm_kernel(
    x_ptr,
    x_pre_ptr,
    pos_ptr,
    pos_pre_ptr,
    w_ptr,
    b_ptr,
    out_pos_ptr,
    out_norm_ptr,
    B,
    n_first_blocks,
    eps,
    L0: tl.constexpr,
    L1: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_POS_PRE: tl.constexpr,
    RETURN_ALL_POS: tl.constexpr,
):
    """Split-grid fused positional add + sequence concat + LayerNorm.

    Programs ``0 .. n_first_blocks - 1`` process ``x`` (``B * L0`` rows) and programs ``n_first_blocks ..`` process
    ``x_pre`` (``B * L1`` rows), giving 100% coalesced reads and zero branch divergence."""
    pid = tl.program_id(0)
    col = tl.arange(0, BLOCK_N)
    col_ok = col < C
    L_TOT: tl.constexpr = L0 + L1
    w = tl.load(w_ptr + col, mask=col_ok, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=col_ok, other=0.0).to(tl.float32)

    if pid < n_first_blocks:
        row = pid.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = row < (B.to(tl.int64) * L0)
        mask = row_ok[:, None] & col_ok[None, :]
        b_idx = row // L0
        l_idx = row - b_idx * L0
        vx = tl.load(x_ptr + row[:, None] * C + col[None, :], mask=mask, other=0.0)
        vp = tl.load(pos_ptr + l_idx[:, None] * C + col[None, :], mask=mask, other=0.0)
        val = (vx.to(tl.float32) + vp.to(tl.float32)).to(out_norm_ptr.dtype.element_ty)
        norm_row = b_idx * L_TOT + l_idx
        if RETURN_ALL_POS:
            tl.store(out_pos_ptr + norm_row[:, None] * C + col[None, :], val, mask=mask)
        else:
            tl.store(out_pos_ptr + row[:, None] * C + col[None, :], val, mask=mask)
    else:
        row = (pid - n_first_blocks).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = row < (B.to(tl.int64) * L1)
        mask = row_ok[:, None] & col_ok[None, :]
        b_idx = row // L1
        l_idx = row - b_idx * L1
        vx = tl.load(x_pre_ptr + row[:, None] * C + col[None, :], mask=mask, other=0.0)
        if HAS_POS_PRE:
            vp = tl.load(pos_pre_ptr + l_idx[:, None] * C + col[None, :], mask=mask, other=0.0)
            val = (vx.to(tl.float32) + vp.to(tl.float32)).to(out_norm_ptr.dtype.element_ty)
        else:
            val = vx.to(out_norm_ptr.dtype.element_ty)
        norm_row = b_idx * L_TOT + L0 + l_idx
        if RETURN_ALL_POS:
            tl.store(out_pos_ptr + norm_row[:, None] * C + col[None, :], val, mask=mask)

    vf = val.to(tl.float32)
    mean = tl.sum(vf, axis=1) / C
    diff = tl.where(col_ok[None, :], vf - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / C
    rstd = tl.rsqrt(var + tl.cast(eps, tl.float32))
    y = ((diff * rstd[:, None]) * w[None, :] + b[None, :]).to(out_norm_ptr.dtype.element_ty)
    tl.store(out_norm_ptr + norm_row[:, None] * C + col[None, :], y, mask=mask)


def pos_cat_layernorm(
    x: Tensor,
    x_pre: Tensor,
    pos: Tensor,
    pos_pre: Tensor | None,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
    *,
    return_all_pos: bool = False,
) -> tuple[Tensor, Tensor]:
    b_size, l0, c = x.shape
    l1 = x_pre.shape[1]
    l_tot = l0 + l1
    if b_size == 0:
        out_pos = x.new_empty((0, l_tot if return_all_pos else l0, c))
        return out_pos, x.new_empty((0, l_tot, c))
    x_c = x.contiguous()
    xp_c = x_pre.contiguous()
    pos_c = pos.to(x.dtype).reshape(l0, c).contiguous()
    pos_pre_c = None if pos_pre is None else pos_pre.to(x.dtype).reshape(l1, c).contiguous()
    out_pos = x.new_empty((b_size, l_tot if return_all_pos else l0, c))
    out_norm = x.new_empty((b_size, l_tot, c))
    block_n = triton.next_power_of_2(c)
    block_m = 8 if block_n <= 256 else 4
    n_first = triton.cdiv(b_size * l0, block_m)
    n_second = triton.cdiv(b_size * l1, block_m)
    _pos_cat_layernorm_kernel[(n_first + n_second,)](
        x_c,
        xp_c,
        pos_c,
        pos_pre_c if pos_pre_c is not None else pos_c,
        weight,
        bias,
        out_pos,
        out_norm,
        b_size,
        n_first,
        eps,
        L0=l0,
        L1=l1,
        C=c,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HAS_POS_PRE=pos_pre_c is not None,
        RETURN_ALL_POS=return_all_pos,
        num_warps=4,
    )
    return out_pos, out_norm


def _pos_cat_ln_eligible(
    x: Tensor,
    x_pre: Tensor,
    pos: Tensor,
    pos_pre: Tensor | None,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
    *,
    return_all_pos: bool = False,
) -> bool:
    return (
        not torch.is_grad_enabled()
        and x.ndim == 3
        and x_pre.ndim == 3
        and x.shape[0] == x_pre.shape[0]
        and x.shape[2] == x_pre.shape[2]
        and x.dtype in _FLOATS
        and x_pre.dtype == x.dtype
        and weight.dtype == x.dtype
        and bias.dtype == x.dtype
        and x.shape[2] <= 1024
        and x.shape[1] > 0
        and x_pre.shape[1] > 0
    )


ops.pos_cat_layernorm.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_pos_cat_ln_eligible
)(pos_cat_layernorm)
