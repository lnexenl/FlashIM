"""Fused deformable-conv patch gather + static INT8 quantize (one kernel, no FP32 intermediate)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim.quant.tensor import Int8Tensor


@triton.jit
def _dcn_gather_quantize_kernel(
    x_ptr,
    offset_ptr,
    out_ptr,
    inv_scale,
    C,
    H,
    W,
    OH,
    OW,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_ob,
    stride_oc,
    K: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_pix = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)

    pix = pid_pix * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    pix_mask = pix < OH * OW
    oh = (pix // OW).to(tl.float32)
    ow = (pix % OW).to(tl.float32)

    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C

    x_base = x_ptr + b * stride_xb
    off_base = offset_ptr + b * stride_ob

    for tap in range(K * K):
        kh = tap // K
        kw = tap % K
        # offset is contiguous [B, 2*K*K, OH, OW]: its spatial plane is row-major, so the flat pixel
        # index `pix` (= oh * OW + ow) already is the offset within one channel.
        off_h_ptr = off_base + (2 * tap) * stride_oc + pix
        off_w_ptr = off_base + (2 * tap + 1) * stride_oc + pix
        dy = tl.load(off_h_ptr, mask=pix_mask, other=0.0)
        dx = tl.load(off_w_ptr, mask=pix_mask, other=0.0)

        sample_h = oh - PADDING + kh + dy
        sample_w = ow - PADDING + kw + dx
        h0 = tl.floor(sample_h)
        w0 = tl.floor(sample_w)
        lh = sample_h - h0
        lw = sample_w - w0
        h0i = h0.to(tl.int32)
        w0i = w0.to(tl.int32)

        acc = tl.zeros((BLOCK_C, BLOCK_PIX), dtype=tl.float32)
        for dh in range(2):
            for dw in range(2):
                hi = h0i + dh
                wi = w0i + dw
                valid = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W) & pix_mask
                hi_c = tl.minimum(tl.maximum(hi, 0), H - 1)
                wi_c = tl.minimum(tl.maximum(wi, 0), W - 1)
                weight = tl.where(dh == 0, 1.0 - lh, lh) * tl.where(dw == 0, 1.0 - lw, lw)
                ptrs = x_base + c[:, None] * stride_xc + hi_c[None, :] * stride_xh + wi_c[None, :] * stride_xw
                mask2d = c_mask[:, None] & valid[None, :]
                val = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
                acc += val * weight[None, :]

        scaled = libdevice.rint(acc * inv_scale)  # reciprocal multiply: exact division is ~2x slower
        quantized = tl.minimum(tl.maximum(scaled, -127.0), 127.0).to(tl.int8)
        out_col = c * (K * K) + tap  # [BLOCK_C]
        row = b * (OH * OW) + pix  # [BLOCK_PIX]
        out_ptrs = out_ptr + row[None, :] * (C * K * K) + out_col[:, None]
        tl.store(out_ptrs, quantized, mask=c_mask[:, None] & pix_mask[None, :])


def deform_patches_int8(x: Tensor, offset: Tensor, kernel_size: int, padding: int, scale: Tensor) -> Int8Tensor:
    """``x``: ``[B, C, H, W]`` FP32 (any layout; converted to channels-last so the per-pixel channel gather
    is coalesced — see the module docstring); ``offset``: ``[B, 2 * K * K, OH, OW]`` contiguous FP32.
    Returns an :class:`Int8Tensor` of logical shape ``[B, OH, OW, C * K * K]``."""
    x = x.contiguous(memory_format=torch.channels_last)
    b, c, h, w = x.shape
    oh, ow = offset.shape[-2:]
    out = torch.empty(b * oh * ow, c * kernel_size * kernel_size, device=x.device, dtype=torch.int8)
    scale = scale.reshape(())
    inv_scale = float((1.0 / scale.clamp_min(torch.finfo(torch.float32).eps)).item())
    block_pix, block_c = 32, min(32, triton.next_power_of_2(c))
    grid = (triton.cdiv(oh * ow, block_pix), triton.cdiv(c, block_c), b)
    _dcn_gather_quantize_kernel[grid](
        x,
        offset,
        out,
        inv_scale,
        c,
        h,
        w,
        oh,
        ow,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        offset.stride(0),
        offset.stride(1),
        K=kernel_size,
        PADDING=padding,
        BLOCK_PIX=block_pix,
        BLOCK_C=block_c,
    )
    return Int8Tensor(out.view(b, oh, ow, c * kernel_size * kernel_size), scale.float())
