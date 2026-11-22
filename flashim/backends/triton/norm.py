"""Multi-row Triton LayerNorm kernels fused with residual add, left-concatenation or window positional encoding.

ATen's ``vectorized_layer_norm_kernel`` launches one warp per row and achieves ~25% of DRAM bandwidth on
128..256-channel BF16 tensors (1.88 ms across 32 calls in XoFTR at 480x640). Processing ``BLOCK_M = 8`` rows per
program with 4 warps reaches 84–86% SOL GDDR7 DRAM bandwidth on sm_120 (ncu: ``_residual_layernorm_kernel`` 86.1%,
``_pos_cat_layernorm_kernel`` 84.1%) and fuses the adjacent residual add, concatenation or positional addition."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


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
    rstd = tl.rsqrt(var + eps)
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
    rstd = tl.rsqrt(var + eps)
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
    rstd = tl.rsqrt(var + eps)
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
