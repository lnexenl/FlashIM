"""Fused RoMaV2 refiner input: projected features, warped features, displacement embedding
and local correlation, written once as the BF16 channels-last input of the first conv block.

The correlation window offsets are whole feature pixels, so every tap shares the
bilinear weights of the warp point: the kernel computes ``f_a . f_b`` at the
``(2r+2)^2`` integer corners once and blends them into the ``(2r+1)^2`` taps, instead
of gathering 4 corners per tap. The 2x2 centre corners also give ``f_ba``.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _refiner_input_kernel(
    fa_ptr,
    fb_ptr,
    warp_ptr,
    emb_w_ptr,
    emb_b_ptr,
    scale_ptr,
    out_ptr,
    rows,
    corr_scale,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    E: tl.constexpr,
    EE: tl.constexpr,
    R: tl.constexpr,
    KK: tl.constexpr,
    HAS_CORR: tl.constexpr,
    C_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row < rows
    pix = row % (HEIGHT * WIDTH)
    img_base = row - pix  # first pixel of this image
    y = pix // WIDTH
    x = pix - y * WIDTH
    ch = tl.arange(0, PP)
    cmask = ch < P

    fa = tl.load(fa_ptr + row[:, None] * P + ch[None, :], mask=valid[:, None] & cmask[None, :], other=0.0)
    wx = tl.load(warp_ptr + row * 2, mask=valid, other=0.0)
    wy = tl.load(warp_ptr + row * 2 + 1, mask=valid, other=0.0)
    # grid_sample(align_corners=False): pixel = ((coord + 1) * size - 1) / 2
    px = ((wx + 1.0) * WIDTH - 1.0) * 0.5
    py = ((wy + 1.0) * HEIGHT - 1.0) * 0.5
    x0f = tl.floor(px)
    y0f = tl.floor(py)
    fx = px - x0f
    fy = py - y0f
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    out_row = out_ptr + row[:, None] * C_TOTAL
    store_mask = valid[:, None] & cmask[None, :]
    tl.store(out_row + ch[None, :], fa.to(tl.bfloat16), mask=store_mask)

    fba = tl.zeros((BLOCK_M, PP), dtype=tl.float32)
    k_idx = tl.arange(0, KK)
    corr = tl.zeros((BLOCK_M, KK), dtype=tl.float32)
    SPAN: tl.constexpr = 2 * R + 2
    TAPS: tl.constexpr = 2 * R + 1
    for v in tl.static_range(SPAN):
        cy = y0 + (v - R)
        in_y = valid & (cy >= 0) & (cy < HEIGHT)
        for u in tl.static_range(SPAN):
            cx = x0 + (u - R)
            inb = in_y & (cx >= 0) & (cx < WIDTH)
            fb = tl.load(
                fb_ptr + (img_base + cy * WIDTH + cx)[:, None] * P + ch[None, :],
                mask=inb[:, None] & cmask[None, :],
                other=0.0,
            )
            if v == R:
                if u == R:
                    fba += w00[:, None] * fb
                if u == R + 1:
                    fba += w10[:, None] * fb
            if v == R + 1:
                if u == R:
                    fba += w01[:, None] * fb
                if u == R + 1:
                    fba += w11[:, None] * fb
            if HAS_CORR:
                dot = tl.sum(fa * fb, axis=1) * corr_scale
                # corner (u, v) feeds taps (i, j) with i in {u-1, u}, j in {v-1, v}
                if u < TAPS and v < TAPS:
                    corr += tl.where(k_idx[None, :] == v * TAPS + u, (w00 * dot)[:, None], 0.0)
                if u >= 1 and v < TAPS:
                    corr += tl.where(k_idx[None, :] == v * TAPS + (u - 1), (w10 * dot)[:, None], 0.0)
                if u < TAPS and v >= 1:
                    corr += tl.where(k_idx[None, :] == (v - 1) * TAPS + u, (w01 * dot)[:, None], 0.0)
                if u >= 1 and v >= 1:
                    corr += tl.where(k_idx[None, :] == (v - 1) * TAPS + (u - 1), (w11 * dot)[:, None], 0.0)
    tl.store(out_row + P + ch[None, :], fba.to(tl.bfloat16), mask=store_mask)

    # Displacement embedding: E outputs of a 1x1 conv over (scale * (warp - grid)).
    gx = (2.0 * x.to(tl.float32) + 1.0) / WIDTH - 1.0
    gy = (2.0 * y.to(tl.float32) + 1.0) / HEIGHT - 1.0
    dx = (wx - gx) * tl.load(scale_ptr)
    dy = (wy - gy) * tl.load(scale_ptr + 1)
    e_idx = tl.arange(0, EE)
    emask = e_idx < E
    ew0 = tl.load(emb_w_ptr + e_idx * 2, mask=emask, other=0.0)
    ew1 = tl.load(emb_w_ptr + e_idx * 2 + 1, mask=emask, other=0.0)
    eb = tl.load(emb_b_ptr + e_idx, mask=emask, other=0.0)
    emb = dx[:, None] * ew0[None, :] + dy[:, None] * ew1[None, :] + eb[None, :]
    tl.store(out_row + 2 * P + e_idx[None, :], emb.to(tl.bfloat16), mask=valid[:, None] & emask[None, :])

    if HAS_CORR:
        tl.store(
            out_row + 2 * P + E + k_idx[None, :],
            corr.to(tl.bfloat16),
            mask=valid[:, None] & (k_idx[None, :] < TAPS * TAPS),
        )


def refiner_input(
    f_a: Tensor,
    f_b: Tensor,
    warp: Tensor,
    emb_weight: Tensor,
    emb_bias: Tensor,
    scale_factor: Tensor,
    radius: int,
    *,
    block_m: int | None = None,
    num_warps: int = 4,
) -> Tensor:
    batch, h, w, p = f_a.shape
    e = emb_weight.shape[0]
    has_corr = radius >= 0
    r = radius if has_corr else 0
    k = (2 * r + 1) ** 2 if has_corr else 0
    c_total = 2 * p + e + k
    rows = batch * h * w
    pp = triton.next_power_of_2(p)
    if block_m is None:  # keep ~3 [BLOCK_M, PP] FP32 tiles in registers
        block_m = max(16, min(128, 4096 // pp))
    out = torch.empty((batch, h, w, c_total), device=f_a.device, dtype=torch.bfloat16)
    _refiner_input_kernel[(triton.cdiv(rows, block_m),)](
        f_a,
        f_b,
        warp,
        emb_weight.reshape(e, 2).float().contiguous(),
        emb_bias.float().contiguous(),
        scale_factor.float().contiguous(),
        out,
        rows,
        1.0 / math.sqrt(p),
        HEIGHT=h,
        WIDTH=w,
        P=p,
        PP=pp,
        E=e,
        EE=triton.next_power_of_2(e),
        R=r,
        KK=triton.next_power_of_2(max(k, 1)),
        HAS_CORR=has_corr,
        C_TOTAL=c_total,
        BLOCK_M=block_m,
        num_warps=num_warps,
    )
    return out.permute(0, 3, 1, 2)


@triton.jit
def _softplus(x):
    # torch.nn.functional.softplus(beta=1, threshold=20)
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))


@triton.jit
def _refiner_heads_kernel(
    z_ptr,
    w_ptr,
    b_ptr,
    prev_warp_ptr,
    prev_conf_ptr,
    warp_ptr,
    conf_ptr,
    rows,
    inv_norm_x,
    inv_norm_y,
    C: tl.constexpr,
    PREV_CONF_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row < rows
    ch = tl.arange(0, C)
    z = tl.load(z_ptr + row[:, None] * C + ch[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    # Six FP32 dot products per pixel on CUDA cores: exact FP32 (the cuDNN 1x1 conv this
    # replaces runs in TF32 under torch's default cudnn.allow_tf32) and cheaper than tl.dot
    # padding 6 outputs to 16.
    d0 = tl.sum(z * tl.load(w_ptr + 0 * C + ch)[None, :], axis=1) + tl.load(b_ptr + 0)
    d1 = tl.sum(z * tl.load(w_ptr + 1 * C + ch)[None, :], axis=1) + tl.load(b_ptr + 1)
    c0 = tl.sum(z * tl.load(w_ptr + 2 * C + ch)[None, :], axis=1) + tl.load(b_ptr + 2)
    x1 = tl.sum(z * tl.load(w_ptr + 3 * C + ch)[None, :], axis=1) + tl.load(b_ptr + 3)
    l10 = tl.sum(z * tl.load(w_ptr + 4 * C + ch)[None, :], axis=1) + tl.load(b_ptr + 4)
    x3 = tl.sum(z * tl.load(w_ptr + 5 * C + ch)[None, :], axis=1) + tl.load(b_ptr + 5)
    warp_x = tl.load(prev_warp_ptr + row * 2, mask=valid) + d0 * inv_norm_x
    warp_y = tl.load(prev_warp_ptr + row * 2 + 1, mask=valid) + d1 * inv_norm_y
    tl.store(warp_ptr + row * 2, warp_x, mask=valid)
    tl.store(warp_ptr + row * 2 + 1, warp_y, mask=valid)

    l00 = _softplus(x1) + 1e-6
    l11 = _softplus(x3) + 1e-6
    p0 = tl.load(prev_conf_ptr + row * PREV_CONF_C, mask=valid)
    tl.store(conf_ptr + row * 4, p0 + c0, mask=valid)
    if PREV_CONF_C == 4:
        tl.store(conf_ptr + row * 4 + 1, tl.load(prev_conf_ptr + row * 4 + 1, mask=valid) + l00 * l00, mask=valid)
        tl.store(conf_ptr + row * 4 + 2, tl.load(prev_conf_ptr + row * 4 + 2, mask=valid) + l00 * l10, mask=valid)
        tl.store(
            conf_ptr + row * 4 + 3,
            tl.load(prev_conf_ptr + row * 4 + 3, mask=valid) + (l10 * l10 + l11 * l11),
            mask=valid,
        )
    else:
        tl.store(conf_ptr + row * 4 + 1, l00 * l00, mask=valid)
        tl.store(conf_ptr + row * 4 + 2, l00 * l10, mask=valid)
        tl.store(conf_ptr + row * 4 + 3, l10 * l10 + l11 * l11, mask=valid)


def refiner_heads(
    z: Tensor,
    weight: Tensor,
    bias: Tensor,
    prev_warp: Tensor,
    prev_confidence: Tensor,
    refine_init: float,
    *,
    block_m: int | None = None,
    num_warps: int = 4,
) -> tuple[Tensor, Tensor]:
    batch, channels, h, w = z.shape
    rows = batch * h * w
    if block_m is None:
        block_m = max(16, min(128, 8192 // channels))
    prev_warp = prev_warp.float().contiguous()
    prev_confidence = prev_confidence.float().contiguous()
    warp = torch.empty((batch, h, w, 2), device=z.device, dtype=torch.float32)
    confidence = torch.empty((batch, h, w, 4), device=z.device, dtype=torch.float32)
    _refiner_heads_kernel[(triton.cdiv(rows, block_m),)](
        z,
        weight.float().contiguous(),
        bias.float().contiguous(),
        prev_warp,
        prev_confidence,
        warp,
        confidence,
        rows,
        1.0 / (refine_init * w),
        1.0 / (refine_init * h),
        C=channels,
        PREV_CONF_C=prev_confidence.shape[-1],
        BLOCK_M=block_m,
        num_warps=num_warps,
    )
    return warp, confidence
