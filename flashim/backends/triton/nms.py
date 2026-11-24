"""Triton kernels for strict window NMS (``ops.strict_nms_mask``) and two-round fast NMS (``ops.simple_nms``)."""

from __future__ import annotations

import torch
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
