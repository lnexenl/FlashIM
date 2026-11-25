"""RoMa (v1)-specific Triton kernels: fused refiner input assembly and FP32 3-channel output head.

``roma_refiner_input`` replaces ``grid_sample`` + ``disp_emb`` + ``local_corr`` + ``torch.cat`` + ``F.pad`` +
``to(bf16)`` + ``to(channels_last)`` + symmetric ``f_b`` copy with one pass (two kernels when ``radius >= 0``:
feature/warp/disp_emb/zero-pad + in-place BF16 local correlation directly into the NHWC output slice).

``roma_refiner_head`` reads the BF16 channels-last refiner output directly (skipping zero-padded channels) and
accumulates the 3 FP32 output channels ``(delta_flow[2], delta_certainty[1])`` without materializing ``d.float()``.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.ops.correlation import local_window
from flashim.platforms import Feature


@triton.jit
def _roma_refiner_input_kernel(
    fa_ptr,
    fb_ptr,
    warp_ptr,
    emb_w_ptr,
    emb_b_ptr,
    out_ptr,
    rows,
    disp_scale,
    swb,
    swc,
    swh,
    sww,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    P: tl.constexpr,
    E: tl.constexpr,
    EE: tl.constexpr,
    C_ORIG: tl.constexpr,
    C_PAD: tl.constexpr,
    PAD_LEN: tl.constexpr,
    PAD_POW2: tl.constexpr,
    PAIR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row < rows
    hw = HEIGHT * WIDTH
    img = row // hw
    pix = row - img * hw
    y = pix // WIDTH
    x = pix - y * WIDTH

    img_b = ((img + PAIR) % (2 * PAIR)) if PAIR > 0 else img
    img_b_base = img_b * hw

    w_off = img * swb + y * swh + x * sww
    wx = tl.load(warp_ptr + w_off, mask=valid, other=0.0)
    wy = tl.load(warp_ptr + w_off + swc, mask=valid, other=0.0)

    # grid_sample(align_corners=False, padding_mode="zeros"): pixel = ((coord + 1) * size - 1) / 2
    px = ((wx + 1.0) * WIDTH - 1.0) * 0.5
    py = ((wy + 1.0) * HEIGHT - 1.0) * 0.5
    x0f = tl.floor(px)
    y0f = tl.floor(py)
    fx = px - x0f
    fy = py - y0f
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    ch = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = ch < P
    store_mask = valid[:, None] & cmask[None, :]
    out_row = out_ptr + row[:, None].to(tl.int64) * C_PAD

    fa = tl.load(fa_ptr + row[:, None].to(tl.int64) * P + ch[None, :], mask=store_mask, other=0.0)
    tl.store(out_row + ch[None, :], fa.to(tl.bfloat16), mask=store_mask)

    in_y0 = valid & (y0 >= 0) & (y0 < HEIGHT)
    in_y1 = valid & (y1 >= 0) & (y1 < HEIGHT)
    in_x0 = (x0 >= 0) & (x0 < WIDTH)
    in_x1 = (x1 >= 0) & (x1 < WIDTH)

    f00 = tl.load(
        fb_ptr + (img_b_base + y0 * WIDTH + x0)[:, None].to(tl.int64) * P + ch[None, :],
        mask=(in_y0 & in_x0)[:, None] & cmask[None, :],
        other=0.0,
    )
    f10 = tl.load(
        fb_ptr + (img_b_base + y0 * WIDTH + x1)[:, None].to(tl.int64) * P + ch[None, :],
        mask=(in_y0 & in_x1)[:, None] & cmask[None, :],
        other=0.0,
    )
    f01 = tl.load(
        fb_ptr + (img_b_base + y1 * WIDTH + x0)[:, None].to(tl.int64) * P + ch[None, :],
        mask=(in_y1 & in_x0)[:, None] & cmask[None, :],
        other=0.0,
    )
    f11 = tl.load(
        fb_ptr + (img_b_base + y1 * WIDTH + x1)[:, None].to(tl.int64) * P + ch[None, :],
        mask=(in_y1 & in_x1)[:, None] & cmask[None, :],
        other=0.0,
    )
    fba = w00[:, None] * f00 + w10[:, None] * f10 + w01[:, None] * f01 + w11[:, None] * f11
    tl.store(out_row + P + ch[None, :], fba.to(tl.bfloat16), mask=store_mask)

    if pid_c == 0:
        gx = (2.0 * x.to(tl.float32) + 1.0) / WIDTH - 1.0
        gy = (2.0 * y.to(tl.float32) + 1.0) / HEIGHT - 1.0
        dx = (wx - gx) * disp_scale
        dy = (wy - gy) * disp_scale
        e_idx = tl.arange(0, EE)
        emask = e_idx < E
        ew0 = tl.load(emb_w_ptr + e_idx * 2, mask=emask, other=0.0)
        ew1 = tl.load(emb_w_ptr + e_idx * 2 + 1, mask=emask, other=0.0)
        eb = tl.load(emb_b_ptr + e_idx, mask=emask, other=0.0)
        emb = dx[:, None] * ew0[None, :] + dy[:, None] * ew1[None, :] + eb[None, :]
        tl.store(out_row + 2 * P + e_idx[None, :], emb.to(tl.bfloat16), mask=valid[:, None] & emask[None, :])
        if PAD_LEN > 0:
            p_idx = tl.arange(0, PAD_POW2)
            pmask = p_idx < PAD_LEN
            zeros = tl.zeros((BLOCK_M, PAD_POW2), dtype=tl.bfloat16)
            tl.store(out_row + C_ORIG + p_idx[None, :], zeros, mask=valid[:, None] & pmask[None, :])


@triton.jit
def _local_corr_into_bf16_kernel(
    fa_ptr,
    fb_ptr,
    warp_ptr,
    window_ptr,
    out_ptr,
    rows,
    scale,
    swb,
    swc,
    swh,
    sww,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    C: tl.constexpr,
    CC: tl.constexpr,
    R: tl.constexpr,
    TT: tl.constexpr,
    C_PAD: tl.constexpr,
    CORR_OFF: tl.constexpr,
    PAIR: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Corner-sharing local correlation writing BF16 directly into ``out[row, CORR_OFF : CORR_OFF + K]``."""
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row < rows
    hw = HEIGHT * WIDTH
    b = row // hw
    pix = row - b * hw
    y = pix // WIDTH
    x = pix - y * WIDTH
    img_b = ((b + PAIR) % (2 * PAIR)) if PAIR > 0 else b
    img_b_base = img_b * hw

    ch = tl.arange(0, CC)
    cmask = ch < C
    f0 = tl.load(fa_ptr + row[:, None].to(tl.int64) * C + ch[None, :], mask=valid[:, None] & cmask[None, :], other=0.0)

    w_off = b * swb + y * swh + x * sww
    wx = tl.load(warp_ptr + w_off, mask=valid, other=0.0)
    wy = tl.load(warp_ptr + w_off + swc, mask=valid, other=0.0)
    x0f = tl.floor(((wx + 1.0) * WIDTH - 1.0) * 0.5)
    y0f = tl.floor(((wy + 1.0) * HEIGHT - 1.0) * 0.5)
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)

    SPAN: tl.constexpr = 2 * R + 2
    TAPS: tl.constexpr = 2 * R + 1
    t = tl.arange(0, TT)
    tmask = t < TAPS
    shift = (t - R).to(tl.float32)[None, :]
    ox = tl.load(window_ptr + t * 2, mask=tmask, other=0.0)
    frac_x = ((wx[:, None] + ox[None, :] + 1.0) * WIDTH - 1.0) * 0.5 - (x0f[:, None] + shift)
    oy = tl.load(window_ptr + t * (2 * TAPS) + 1, mask=tmask, other=0.0)
    frac_y = ((wy[:, None] + oy[None, :] + 1.0) * HEIGHT - 1.0) * 0.5 - (y0f[:, None] + shift)
    out_base = out_ptr + row[:, None].to(tl.int64) * C_PAD + CORR_OFF + t[None, :]
    store_mask = valid[:, None] & tmask[None, :]

    lower = tl.zeros((BLOCK_M, TT), dtype=tl.float32)
    for v in range(SPAN):
        upper = tl.zeros((BLOCK_M, TT), dtype=tl.float32)
        wy_top = 1.0 - tl.sum(tl.where(t[None, :] == v, frac_y, 0.0), axis=1)
        wy_bottom = tl.sum(tl.where(t[None, :] == v - 1, frac_y, 0.0), axis=1)
        cy = y0 + (v - R)
        in_y = valid & (cy >= 0) & (cy < HEIGHT)
        for u in range(SPAN):
            cx = x0 + (u - R)
            inb = in_y & (cx >= 0) & (cx < WIDTH)
            f1 = tl.load(
                fb_ptr + (img_b_base + cy * WIDTH + cx)[:, None].to(tl.int64) * C + ch[None, :],
                mask=inb[:, None] & cmask[None, :],
                other=0.0,
            )
            dot = tl.sum(f0 * f1, axis=1) * scale
            hx = tl.where(t[None, :] == u, 1.0 - frac_x, tl.where(t[None, :] == u - 1, frac_x, 0.0))
            hx = hx * dot[:, None]
            upper += hx * wy_top[:, None]
            lower += hx * wy_bottom[:, None]
        if v >= 1:
            tl.store(out_base + (v - 1) * TAPS, lower.to(tl.bfloat16), mask=store_mask)
        lower = upper


def roma_refiner_input(
    f_a: Tensor,
    f_b: Tensor,
    warp: Tensor,
    emb_weight: Tensor,
    emb_bias: Tensor,
    scale_factor: float,
    radius: int,
    padded_channels: int,
    pair: int = 0,
) -> Tensor:
    batch, p, h, w = f_a.shape
    e = emb_weight.shape[0]
    k = (2 * radius + 1) ** 2 if radius >= 0 else 0
    c_orig = 2 * p + e + k
    c_pad = max(padded_channels, c_orig)
    pad_len = c_pad - c_orig
    rows = batch * h * w

    if f_a.stride(1) != 1:
        f_a = f_a.contiguous(memory_format=torch.channels_last)
    if pair > 0:
        f_b = f_a
    elif f_b.stride(1) != 1:
        f_b = f_b.contiguous(memory_format=torch.channels_last)

    out = torch.empty((batch, h, w, c_pad), device=f_a.device, dtype=torch.bfloat16)
    swb, swc, swh, sww = warp.stride()
    block_c = min(triton.next_power_of_2(p), 64)
    block_m = 32 if block_c >= 64 else 64
    _roma_refiner_input_kernel[(triton.cdiv(rows, block_m), triton.cdiv(p, block_c))](
        f_a,
        f_b,
        warp,
        emb_weight.reshape(e, 2).float().contiguous(),
        emb_bias.float().contiguous(),
        out,
        rows,
        (40.0 / 32.0) * float(scale_factor),
        swb,
        swc,
        swh,
        sww,
        HEIGHT=h,
        WIDTH=w,
        P=p,
        E=e,
        EE=triton.next_power_of_2(e),
        C_ORIG=c_orig,
        C_PAD=c_pad,
        PAD_LEN=pad_len,
        PAD_POW2=triton.next_power_of_2(max(pad_len, 1)),
        PAIR=pair,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=4,
    )
    if radius >= 0:
        corr_bm = 16 if p <= 256 else 8
        window = local_window(radius, h, w, f_a.device)
        _local_corr_into_bf16_kernel[(triton.cdiv(rows, corr_bm),)](
            f_a,
            f_b,
            warp,
            window,
            out,
            rows,
            1.0 / math.sqrt(p),
            swb,
            swc,
            swh,
            sww,
            HEIGHT=h,
            WIDTH=w,
            C=p,
            CC=triton.next_power_of_2(p),
            R=radius,
            TT=triton.next_power_of_2(2 * radius + 1),
            C_PAD=c_pad,
            CORR_OFF=2 * p + e,
            PAIR=pair,
            BLOCK_M=corr_bm,
            num_warps=4,
        )
    return out.permute(0, 3, 1, 2)


def _refiner_input_eligible(
    f_a: Tensor,
    f_b: Tensor,
    warp: Tensor,
    emb_weight: Tensor,
    emb_bias: Tensor,
    scale_factor: float,
    radius: int,
    padded_channels: int,
    pair: int = 0,
) -> bool:
    p = f_a.shape[1]
    e = emb_weight.shape[0]
    k = (2 * radius + 1) ** 2 if radius >= 0 else 0
    pad_len = max(padded_channels - (2 * p + e + k), 0)
    return (
        f_a.ndim == 4
        and f_a.dtype == torch.float32
        and f_b.dtype == torch.float32
        and warp.ndim == 4
        and warp.shape[1] == 2
        and warp.dtype == torch.float32
        and 1 <= p <= 512
        and 1 <= e <= 128
        and -1 <= radius <= 7
        and pad_len <= 64
        and (pair == 0 or (pair > 0 and f_a.shape[0] == 2 * pair))
    )


ops.roma_refiner_input.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_refiner_input_eligible
)(roma_refiner_input)


# -- refiner head: BF16 NHWC -> FP32 (delta_flow [B, 2, H, W], delta_certainty [B, 1, H, W]) ----


@triton.jit
def _roma_refiner_head_kernel(
    z_ptr,
    w_ptr,
    b_ptr,
    flow_ptr,
    cert_ptr,
    rows,
    HW: tl.constexpr,
    C_ORIG: tl.constexpr,
    C_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row < rows
    img = row // HW
    pix = row - img * HW

    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32) + tl.load(b_ptr + 0)
    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32) + tl.load(b_ptr + 1)
    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32) + tl.load(b_ptr + 2)

    z_row = z_ptr + row[:, None].to(tl.int64) * C_PAD
    for c0 in range(0, C_ORIG, BLOCK_C):
        ch = c0 + tl.arange(0, BLOCK_C)
        cmask = ch < C_ORIG
        z = tl.load(z_row + ch[None, :], mask=valid[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        w0 = tl.load(w_ptr + 0 * C_ORIG + ch, mask=cmask, other=0.0)
        w1 = tl.load(w_ptr + 1 * C_ORIG + ch, mask=cmask, other=0.0)
        w2 = tl.load(w_ptr + 2 * C_ORIG + ch, mask=cmask, other=0.0)
        acc0 += tl.sum(z * w0[None, :], axis=1)
        acc1 += tl.sum(z * w1[None, :], axis=1)
        acc2 += tl.sum(z * w2[None, :], axis=1)

    f_base = (img * 2 * HW + pix).to(tl.int64)
    tl.store(flow_ptr + f_base, acc0, mask=valid)
    tl.store(flow_ptr + f_base + HW, acc1, mask=valid)
    tl.store(cert_ptr + row.to(tl.int64), acc2, mask=valid)


def roma_refiner_head(z: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    batch, c_pad, h, w = z.shape
    c_orig = weight.shape[1]
    if not z.is_contiguous(memory_format=torch.channels_last):
        z = z.contiguous(memory_format=torch.channels_last)
    rows = batch * h * w
    delta_flow = torch.empty((batch, 2, h, w), device=z.device, dtype=torch.float32)
    delta_cert = torch.empty((batch, 1, h, w), device=z.device, dtype=torch.float32)
    block_c = min(triton.next_power_of_2(c_orig), 128)
    block_m = 32 if block_c >= 64 else 64
    _roma_refiner_head_kernel[(triton.cdiv(rows, block_m),)](
        z,
        weight.float().contiguous(),
        bias.float().contiguous(),
        delta_flow,
        delta_cert,
        rows,
        HW=h * w,
        C_ORIG=c_orig,
        C_PAD=c_pad,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return delta_flow, delta_cert


def _refiner_head_eligible(z: Tensor, weight: Tensor, bias: Tensor) -> bool:
    return (
        z.ndim == 4
        and z.dtype in (torch.bfloat16, torch.float32)
        and weight.ndim == 2
        and weight.shape[0] == 3
        and 1 <= weight.shape[1] <= z.shape[1]
        and tuple(bias.shape) == (3,)
    )


ops.roma_refiner_head.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_refiner_head_eligible
)(roma_refiner_head)
