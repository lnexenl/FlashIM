"""Fused DUSt3R DPT post-processing (``ops.dust3r_pts3d_conf``): reads NHWC FP16/BF16/FP32 ``[B, 4, H, W]``
once and writes FP32 ``pts3d [B, H, W, 3]`` and ``conf [B, H, W]`` in one pass."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


@triton.jit
def _dust3r_pts3d_conf_kernel(
    raw_ptr,
    pts_ptr,
    conf_ptr,
    n_pix,
    conf_offset,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pix
    base4 = offs.to(tl.int64) * 4
    x = tl.load(raw_ptr + base4 + 0, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(raw_ptr + base4 + 1, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(raw_ptr + base4 + 2, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(raw_ptr + base4 + 3, mask=mask, other=0.0).to(tl.float32)

    d = tl.sqrt(x * x + y * y + z * z)
    scale = libdevice.expm1(d) / tl.maximum(d, 1e-8)

    base3 = offs.to(tl.int64) * 3
    tl.store(pts_ptr + base3 + 0, x * scale, mask=mask)
    tl.store(pts_ptr + base3 + 1, y * scale, mask=mask)
    tl.store(pts_ptr + base3 + 2, z * scale, mask=mask)
    tl.store(conf_ptr + offs, conf_offset + tl.exp(c), mask=mask)


def dust3r_pts3d_conf(raw: Tensor, conf_offset: float = 1.0) -> tuple[Tensor, Tensor]:
    b, _, h, w = raw.shape
    n_pix = b * h * w
    pts = torch.empty((b, h, w, 3), device=raw.device, dtype=torch.float32)
    conf = torch.empty((b, h, w), device=raw.device, dtype=torch.float32)
    block = 256
    _dust3r_pts3d_conf_kernel[(triton.cdiv(n_pix, block),)](
        raw, pts, conf, n_pix, float(conf_offset), BLOCK=block, num_warps=4
    )
    return pts, conf


def _eligible(raw: Tensor, conf_offset: float = 1.0) -> bool:
    del conf_offset
    return (
        raw.is_cuda
        and raw.ndim == 4
        and raw.shape[1] == 4
        and raw.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and raw.is_contiguous(memory_format=torch.channels_last)
        and raw.numel() < 2**31
    )


ops.dust3r_pts3d_conf.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible
)(dust3r_pts3d_conf)
