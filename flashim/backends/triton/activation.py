"""Fused channel-wise ``relu(x) * scale + shift`` in Triton for NHWC (channels-last) 4D tensors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


@triton.jit
def _relu_affine_nhwc_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    N_PIX,
    C: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
):
    pid = tl.program_id(0)
    pix = pid * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    pix_ok = pix < N_PIX
    ch = tl.arange(0, C)
    scale = tl.load(scale_ptr + ch).to(tl.float32)
    shift = tl.load(shift_ptr + ch).to(tl.float32)
    offs = pix[:, None].to(tl.int64) * C + ch[None, :]
    x = tl.load(x_ptr + offs, mask=pix_ok[:, None], other=0.0).to(tl.float32)
    y = tl.maximum(x, 0.0) * scale[None, :] + shift[None, :]
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=pix_ok[:, None])


def relu_affine_triton(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    b, c, h, w = x.shape
    out = torch.empty_like(x)
    n_pix = b * h * w
    block_pix = 32 if c <= 128 else 16
    grid = (triton.cdiv(n_pix, block_pix),)
    _relu_affine_nhwc_kernel[grid](
        x, scale, shift, out, n_pix, C=c, BLOCK_PIX=block_pix, num_warps=4
    )
    return out


def _eligible(x: Tensor, scale: Tensor, shift: Tensor) -> bool:
    c = x.shape[1] if x.ndim == 4 else 0
    return (
        x.ndim == 4
        and x.is_contiguous(memory_format=torch.channels_last)
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and scale.is_contiguous()
        and shift.is_contiguous()
        and scale.dtype == torch.float32
        and shift.dtype == torch.float32
        and c in (32, 64, 128, 256)
    )


@ops.relu_affine.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _relu_affine_impl(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return relu_affine_triton(x, scale, shift)
