"""Triton kernels for strict window NMS (``ops.strict_nms_mask``) and iterative 2D box NMS (``ops.box_nms_2d``)."""

from __future__ import annotations

import torch
import torchvision.ops
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


@triton.jit
def _strict_nms_mask_kernel(
    hm_ptr, out_ptr, H, W, s_b, s_h, s_w, threshold,
    R: tl.constexpr, HAS_THRESH: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    oy = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ox = (tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    b = tl.program_id(2).to(tl.int64)
    in_map = (oy < H) & (ox < W)
    in_inner = (oy >= R) & (oy < H - R) & (ox >= R) & (ox < W - R)

    base = hm_ptr + b * s_b
    c = tl.load(base + oy.to(tl.int64) * s_h + ox.to(tl.int64) * s_w, mask=in_inner, other=0.0)
    keep = in_inner & (c > threshold) if HAS_THRESH else in_inner

    for dy in tl.static_range(-R, R + 1):
        for dx in tl.static_range(-R, R + 1):
            if dy != 0 or dx != 0:
                ny = oy + dy
                nx = ox + dx
                nb = tl.load(base + ny.to(tl.int64) * s_h + nx.to(tl.int64) * s_w, mask=in_inner, other=0.0)
                keep = keep & (c > nb)

    out_off = (b * H + oy.to(tl.int64)) * W + ox.to(tl.int64)
    tl.store(out_ptr + out_off, keep, mask=in_map)


def _strict_nms_mask_triton(heatmap: Tensor, window: int = 5, threshold: float | None = None) -> Tensor:
    b, _, h, w = heatmap.shape
    r = window // 2
    if h <= 2 * r or w <= 2 * r:
        return torch.zeros_like(heatmap, dtype=torch.bool)
    out = torch.empty(heatmap.shape, device=heatmap.device, dtype=torch.bool)
    block_h, block_w = 16, 64
    grid = (triton.cdiv(w, block_w), triton.cdiv(h, block_h), b)
    _strict_nms_mask_kernel[grid](
        heatmap, out, h, w, heatmap.stride(0), heatmap.stride(2), heatmap.stride(3),
        0.0 if threshold is None else float(threshold),
        R=r, HAS_THRESH=threshold is not None, BLOCK_H=block_h, BLOCK_W=block_w, num_warps=4,
    )
    return out


def _strict_nms_eligible(heatmap: Tensor, window: int = 5, threshold: float | None = None) -> bool:
    return (
        heatmap.is_cuda and heatmap.ndim == 4 and heatmap.shape[1] == 1
        and window % 2 == 1 and 1 <= window <= 9
        and heatmap.dtype in (torch.float32, torch.bfloat16, torch.float16)
    )


ops.strict_nms_mask.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_strict_nms_eligible
)(_strict_nms_mask_triton)


@triton.jit
def _mark_kept_4x4_iou01_kernel(
    scores_ptr,
    state_ptr,
    out_state_ptr,
    H,
    W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    py = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    px = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_bounds = (py[:, None] < H) & (px[None, :] < W)
    idx = py[:, None] * W + px[None, :]
    st = tl.load(state_ptr + idx, mask=in_bounds, other=0)
    is_active = in_bounds & (st == 1)
    if tl.max(is_active.to(tl.int32)) == 0:
        tl.store(out_state_ptr + idx, st, mask=in_bounds)
        return

    val = tl.load(scores_ptr + idx, mask=is_active, other=-1.0)
    strict_max = is_active

    for dy in tl.static_range(-3, 4):
        ny = py + dy
        y_ok = (ny >= 0) & (ny < H)
        for dx in tl.static_range(-3, 4):
            # Compile-time exact 4x4 IoU > 0.1 stencil: (4 - |dy|) * (4 - |dx|) >= 3, excluding (0, 0)
            if (dy != 0 or dx != 0) and (4 - (dy if dy >= 0 else -dy)) * (4 - (dx if dx >= 0 else -dx)) >= 3:
                nx = px + dx
                n_ok = y_ok[:, None] & (nx[None, :] >= 0) & (nx[None, :] < W)
                n_idx = ny[:, None] * W + nx[None, :]
                n_st = tl.load(state_ptr + n_idx, mask=n_ok, other=0)
                n_act = n_ok & (n_st == 1)
                n_val = tl.load(scores_ptr + n_idx, mask=n_act, other=-1.0)
                strict_max = strict_max & (~n_act | (val > n_val))

    new_st = tl.where(strict_max, 2, st).to(tl.uint8)
    tl.store(out_state_ptr + idx, new_st, mask=in_bounds)


@triton.jit
def _suppress_4x4_iou01_kernel(
    state_ptr,
    out_state_ptr,
    H,
    W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    py = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    px = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_bounds = (py[:, None] < H) & (px[None, :] < W)
    idx = py[:, None] * W + px[None, :]
    st = tl.load(state_ptr + idx, mask=in_bounds, other=0)
    is_active = in_bounds & (st == 1)
    if tl.max(is_active.to(tl.int32)) == 0:
        tl.store(out_state_ptr + idx, st, mask=in_bounds)
        return

    suppressed = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int1)
    for dy in tl.static_range(-3, 4):
        ny = py + dy
        y_ok = (ny >= 0) & (ny < H)
        for dx in tl.static_range(-3, 4):
            if (dy != 0 or dx != 0) and (4 - (dy if dy >= 0 else -dy)) * (4 - (dx if dx >= 0 else -dx)) >= 3:
                nx = px + dx
                n_ok = y_ok[:, None] & (nx[None, :] >= 0) & (nx[None, :] < W)
                n_idx = ny[:, None] * W + nx[None, :]
                n_st = tl.load(state_ptr + n_idx, mask=n_ok, other=0)
                suppressed = suppressed | (n_ok & (n_st == 2))

    new_st = tl.where(is_active & suppressed, 0, st).to(tl.uint8)
    tl.store(out_state_ptr + idx, new_st, mask=in_bounds)


def box_nms_2d_triton(
    scores_hw: Tensor, threshold: float, size: int = 4, iou_threshold: float = 0.1, rounds: int = 4
) -> tuple[Tensor, Tensor]:
    scores = scores_hw.float().contiguous()
    h, w = scores.shape
    state = (scores >= threshold).to(torch.uint8)
    scratch = torch.empty_like(state)
    block_h, block_w = 16, 32
    grid = (triton.cdiv(h, block_h), triton.cdiv(w, block_w))

    for _ in range(rounds):
        _mark_kept_4x4_iou01_kernel[grid](
            scores, state, scratch, h, w, BLOCK_H=block_h, BLOCK_W=block_w, num_warps=4
        )
        _suppress_4x4_iou01_kernel[grid](
            scratch, state, h, w, BLOCK_H=block_h, BLOCK_W=block_w, num_warps=4
        )

    ky, kx = torch.where(state == 2)
    ry, rx = torch.where(state == 1)
    kpts_k = torch.stack((kx.float(), ky.float()), dim=-1)
    vals_k = scores[ky, kx]

    if ry.numel() == 0:
        return kpts_k, vals_k

    # Resolve any tiny undecided remainder (chains > rounds or exact score ties) with exact torchvision NMS.
    r_vals = scores[ry, rx]
    half = size / 2.0
    ryf, rxf = ry.float(), rx.float()
    boxes = torch.stack((ryf - half, rxf - half, ryf + half, rxf + half), dim=-1)
    keep = torchvision.ops.nms(boxes, r_vals, iou_threshold)
    kpts_r = torch.stack((rxf[keep], ryf[keep]), dim=-1)
    vals_r = r_vals[keep]
    if ky.numel() == 0:
        return kpts_r, vals_r
    return torch.cat((kpts_k, kpts_r), dim=0), torch.cat((vals_k, vals_r), dim=0)


def _eligible(scores_hw: Tensor, threshold: float, size: int = 4, iou_threshold: float = 0.1) -> bool:
    return (
        scores_hw.ndim == 2
        and scores_hw.numel() > 0
        and size == 4
        and abs(iou_threshold - 0.1) < 1e-6
    )


@ops.box_nms_2d.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _box_nms_2d_impl(
    scores_hw: Tensor, threshold: float, size: int = 4, iou_threshold: float = 0.1
) -> tuple[Tensor, Tensor]:
    return box_nms_2d_triton(scores_hw, threshold, size, iou_threshold)
