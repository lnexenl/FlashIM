"""Depthwise 5x5 conv-block kernels (ported from RoMaV2 ``refiner_fused_mxfp8.py``), and fused implicit-GEMM
convolutions for a full ``groups == 1`` ``nn.Conv2d``.

- ``depthwise5x5_relu_mxfp8``: depthwise + bias + ReLU -> dynamic per-(row, 32-channel)
  MXFP8, scales written directly in the ``SWIZZLE_32_4_4`` layout ``F.scaled_mm`` expects;
- ``depthwise5x5_relu_int8``: the same producer with a static per-tensor INT8 output;
- ``depthwise5x5_relu_pointwise``: BF16 depthwise + ReLU + pointwise 1x1 in one pass;
- ``depthwise5x5_relu_pointwise_mxfp8``: the MXFP8 variant with the pointwise GEMM fused;
- ``int8_conv2d`` / ``mxfp8_conv2d``: one tap per ``tl.dot`` (INT8) or per-32-channel-block ``tl.dot`` (MXFP8),
  gathered straight from the activation; no materialized im2col buffer.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim import ops
from flashim.backends.triton.quantize import to_e4m3
from flashim.ops._registry import DeviceSpec
from flashim.ops.conv import (
    BLOCK_SCALED_CONV_CAPABILITY,
    DEPTHWISE5X5_RELU_POINTWISE_CHANNELS,
    fused_mxfp8_conv2d_worthwhile,
)
from flashim.platforms import Feature, platform_for


@triton.jit
def _load_tap_row(input_ptr, row_base, x, row_ok, x_ok, channel, CHANNELS: tl.constexpr, WIDTH: tl.constexpr,
                  KX: tl.constexpr):
    """One horizontally shifted input row ``[BW, len(channel)]`` (zero padded), rounded to BF16."""
    input_x = x + (KX - 2)
    mask = x_ok & row_ok & (input_x >= 0) & (input_x < WIDTH)
    values = tl.load(input_ptr + (row_base + input_x)[:, None] * CHANNELS + channel[None, :], mask=mask[:, None],
                     other=0.0)
    return values.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _depthwise5x5_relu_mxfp8_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    scale_bits_ptr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    GROUPS: tl.constexpr,
    SCALE_TILES: tl.constexpr,
    BW: tl.constexpr,
    TILE_H: tl.constexpr,
    INT8: tl.constexpr,
    GROUPS_FIRST: tl.constexpr = False,
):
    """Depthwise 5x5 + bias + ReLU + dynamic MXFP8 (or, ``INT8``, static per-tensor INT8: ``scale_bits_ptr``
    is then the FP32 scale and the ``float8`` output is ``int8``, activations rounded to BF16 first) over a ``BW x TILE_H`` tile, 32 channels.

    Row rolling: each input row's 5 horizontal shifts are loaded once and accumulated into
    the 5 output rows that use it (5 loads per pixel instead of 25). Per output pixel the
    FP32 accumulation order is ky-major, kx-minor, as in a direct 25-tap loop.
    """
    if GROUPS_FIRST:  # grid (groups, x tiles, batch * y tiles): the channel groups of one tile are scheduled together
        Y_TILES: tl.constexpr = (HEIGHT + TILE_H - 1) // TILE_H
        group = tl.program_id(0)
        x = tl.program_id(1) * BW + tl.arange(0, BW)
        y_start = (tl.program_id(2) % Y_TILES) * TILE_H
        batch = tl.program_id(2) // Y_TILES
    else:
        x = tl.program_id(0) * BW + tl.arange(0, BW)
        y_start = tl.program_id(1) * TILE_H
        batch = tl.program_id(2) // GROUPS
        group = tl.program_id(2) % GROUPS
    x_ok = x < WIDTH
    channel = group * 32 + tl.arange(0, 32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    acc0 = tl.zeros((BW, 32), dtype=tl.float32)
    acc1 = tl.zeros((BW, 32), dtype=tl.float32)
    acc2 = tl.zeros((BW, 32), dtype=tl.float32)
    acc3 = tl.zeros((BW, 32), dtype=tl.float32)
    acc4 = tl.zeros((BW, 32), dtype=tl.float32)
    for step in range(TILE_H + 4):
        input_y = y_start - 2 + step
        row_ok = (input_y >= 0) & (input_y < HEIGHT)
        row_base = (batch * HEIGHT + input_y) * WIDTH
        for kx in tl.static_range(5):
            values = _load_tap_row(input_ptr, row_base, x, row_ok, x_ok, channel, CHANNELS, WIDTH, kx)
            # output row input_y + 2 - ky receives tap ky: acc4 <- ky0 ... acc0 <- ky4
            acc4 += values * tl.load(weight_ptr + (0 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc3 += values * tl.load(weight_ptr + (1 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc2 += values * tl.load(weight_ptr + (2 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc1 += values * tl.load(weight_ptr + (3 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc0 += values * tl.load(weight_ptr + (4 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
        out_y = input_y - 2
        if (step >= 4) & (out_y < HEIGHT):  # acc0 now holds all 25 taps of row out_y
            activated = tl.maximum(acc0 + bias[None, :], 0.0)
            row = (batch * HEIGHT + out_y) * WIDTH + x
            if INT8:
                scaled = libdevice.rint(activated.to(tl.bfloat16).to(tl.float32) * (1.0 / tl.load(scale_bits_ptr)))
                quantized = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
                tl.store(output_ptr + row[:, None] * CHANNELS + channel[None, :], quantized, mask=x_ok[:, None])
            else:
                amax = tl.max(tl.abs(activated), axis=1)
                exponent = tl.ceil(tl.log2(amax / 448.0))
                exponent = tl.maximum(tl.minimum(exponent, 127.0), -127.0)
                exponent = tl.where(amax == 0.0, -127.0, exponent)
                quantized = tl.maximum(tl.minimum(activated * tl.exp2(-exponent)[:, None], 448.0), -448.0)
                output = output_ptr + row[:, None] * CHANNELS + channel[None, :]
                tl.store(output, to_e4m3(quantized), mask=x_ok[:, None])
                row_in_tile = row % 128
                scale_offset = (
                    ((row // 128) * SCALE_TILES + group // 4) * 512
                    + (row_in_tile % 32) * 16
                    + (row_in_tile // 32) * 4
                    + group % 4
                )
                tl.store(scale_bits_ptr + scale_offset, (exponent + 127.0).to(tl.uint8), mask=x_ok)
        acc0 = acc1
        acc1 = acc2
        acc2 = acc3
        acc3 = acc4
        acc4 = tl.zeros((BW, 32), dtype=tl.float32)


def _depthwise_tile(value: Tensor, bw: int | None, groups_first: bool | None) -> tuple[int, bool]:
    """``(bw, groups_first)`` defaults. On Ada with 512 channels the 16 channel groups of a tile are scheduled together
    (group-major grid) in 8-pixel-wide tiles: ncu on an L20 showed the group-minor order reading 2.4x the input from
    DRAM (each group sweeps the image on its own). Measured there, bit-identical: MXFP8 1x512x256x256 0.295 -> 0.266
    ms, INT8 2x512x256x256 0.444 -> 0.380 ms, 2x512x200x200 0.285 -> 0.247 ms. With 4 groups (128 channels) the same
    order is 6-8% slower, so it stays group-minor; other architectures are unmeasured and keep it too."""
    if groups_first is None:
        groups_first = value.shape[1] >= 512 and platform_for(value.device).capability == (8, 9)
        if bw is None and groups_first:
            bw = 8
    return bw or 16, bool(groups_first)


def _depthwise_grid(batch: int, groups: int, height: int, width: int, bw: int, tile_h: int, groups_first: bool) -> tuple:
    if groups_first:
        return groups, triton.cdiv(width, bw), batch * triton.cdiv(height, tile_h)
    return triton.cdiv(width, bw), triton.cdiv(height, tile_h), batch * groups


def depthwise5x5_relu_int8(
    value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor, *, bw: int | None = None, tile_h: int | None = None,
    num_warps: int = 2, groups_first: bool | None = None,
) -> Tensor:
    """Depthwise 5x5 + bias + ReLU -> INT8 ``[B*H*W, C]`` with one static FP32 ``scale``."""
    batch, channels, height, width = value.shape
    bw, groups_first = _depthwise_tile(value, bw, groups_first)
    tile_h = tile_h or (32 if height > 64 else 16)
    quantized = torch.empty((batch * height * width, channels), device=value.device, dtype=torch.int8)
    _depthwise5x5_relu_mxfp8_kernel[_depthwise_grid(batch, channels // 32, height, width, bw, tile_h, groups_first)](
        value,
        weight,
        bias,
        quantized,
        scale,
        HEIGHT=height,
        WIDTH=width,
        CHANNELS=channels,
        GROUPS=channels // 32,
        SCALE_TILES=0,
        BW=bw,
        TILE_H=tile_h,
        INT8=True,
        GROUPS_FIRST=groups_first,
        num_warps=num_warps,
    )
    return quantized


def depthwise5x5_relu_mxfp8(
    value: Tensor, weight: Tensor, bias: Tensor, *, bw: int | None = None, tile_h: int | None = None,
    num_warps: int = 2, groups_first: bool | None = None,
) -> tuple[Tensor, Tensor]:
    batch, channels, height, width = value.shape
    bw, groups_first = _depthwise_tile(value, bw, groups_first)
    groups = channels // 32
    scale_tiles = (groups + 3) // 4
    rows = batch * height * width
    padded_rows = triton.cdiv(rows, 128) * 128
    tile_h = tile_h or (32 if height > 64 else 16)
    quantized = torch.empty((rows, channels), device=value.device, dtype=torch.float8_e4m3fn)
    scale_bits = torch.zeros(padded_rows * scale_tiles * 4, device=value.device, dtype=torch.uint8)
    _depthwise5x5_relu_mxfp8_kernel[_depthwise_grid(batch, groups, height, width, bw, tile_h, groups_first)](
        value,
        weight,
        bias,
        quantized,
        scale_bits,
        HEIGHT=height,
        WIDTH=width,
        CHANNELS=channels,
        GROUPS=groups,
        SCALE_TILES=scale_tiles,
        BW=bw,
        TILE_H=tile_h,
        INT8=False,
        GROUPS_FIRST=groups_first,
        num_warps=num_warps,
    )
    return quantized, scale_bits.view(torch.float8_e8m0fnu)


@triton.jit
def _depthwise5x5_relu_pointwise_kernel(
    input_ptr,
    dw_weight_ptr,
    dw_bias_ptr,
    pw_weight_ptr,
    pw_bias_ptr,
    output_ptr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    BW: tl.constexpr,
    TILE_H: tl.constexpr,
):
    """BF16 refiner block: depthwise 5x5 + bias + ReLU + CxC pointwise + bias, channels-last.

    Row rolling over a ``BW x TILE_H`` tile (see the MXFP8 depthwise kernel). The depthwise
    output is rounded to BF16, as a BF16 convolution would, before the pointwise ``tl.dot``.
    """
    x = tl.program_id(0) * BW + tl.arange(0, BW)
    y_start = tl.program_id(1) * TILE_H
    batch = tl.program_id(2)
    x_ok = x < WIDTH
    channel = tl.arange(0, CHANNELS)
    bias = tl.load(dw_bias_ptr + channel).to(tl.float32)
    pw_weight = tl.load(pw_weight_ptr + channel[:, None] * CHANNELS + channel[None, :])  # [out, in]
    pw_bias = tl.load(pw_bias_ptr + channel).to(tl.float32)
    acc0 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc1 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc2 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc3 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc4 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    for step in range(TILE_H + 4):
        input_y = y_start - 2 + step
        row_ok = (input_y >= 0) & (input_y < HEIGHT)
        row_base = (batch * HEIGHT + input_y) * WIDTH
        for kx in tl.static_range(5):
            values = _load_tap_row(input_ptr, row_base, x, row_ok, x_ok, channel, CHANNELS, WIDTH, kx)
            acc4 += values * tl.load(dw_weight_ptr + (0 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc3 += values * tl.load(dw_weight_ptr + (1 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc2 += values * tl.load(dw_weight_ptr + (2 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc1 += values * tl.load(dw_weight_ptr + (3 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc0 += values * tl.load(dw_weight_ptr + (4 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
        out_y = input_y - 2
        if (step >= 4) & (out_y < HEIGHT):
            hidden = tl.maximum(acc0 + bias[None, :], 0.0).to(tl.bfloat16)
            out = tl.dot(hidden, tl.trans(pw_weight)) + pw_bias[None, :]
            row = (batch * HEIGHT + out_y) * WIDTH + x
            tl.store(output_ptr + row[:, None] * CHANNELS + channel[None, :], out.to(tl.bfloat16),
                     mask=x_ok[:, None])
        acc0 = acc1
        acc1 = acc2
        acc2 = acc3
        acc3 = acc4
        acc4 = tl.zeros((BW, CHANNELS), dtype=tl.float32)


def depthwise5x5_relu_pointwise(
    value: Tensor,
    dw_weight: Tensor,
    dw_bias: Tensor,
    pw_weight: Tensor,
    pw_bias: Tensor,
    *,
    bw: int = 32,
    tile_h: int | None = None,
    num_warps: int | None = None,
) -> Tensor:
    """``num_warps`` defaults to one warp per 16 channels: the five ``[BW, C]`` FP32 row accumulators must stay in
    registers. Measured on sm_120 (BF16 channels-last, vs cuDNN depthwise + ReLU + cuDNN 1x1): 64 ch at 896x1200
    0.47 ms with 4 warps (35.7 ms with 2: spills) vs 2.06 ms; 128 ch at 448x600 0.32 ms with 8 warps vs 1.04 ms;
    32 ch at 896x1200 0.23 ms with 2 warps vs 1.03 ms; 16 ch 0.115 ms with 1 warp vs 0.44 ms. 256 ch only reaches
    0.36 vs 0.44 ms (bw=16, 16 warps) and 512 ch loses, so those stay on cuDNN (see ``_dw_pw_eligible``)."""
    batch, channels, height, width = value.shape
    if (batch * height * width + bw) * channels >= 2**31:
        raise ValueError("depthwise5x5_relu_pointwise: tensor too large for int32 offsets")
    tile_h = tile_h or (32 if height > 64 else 16)
    num_warps = num_warps or max(1, channels // 16)
    out = torch.empty((batch, height, width, channels), device=value.device, dtype=torch.bfloat16)
    _depthwise5x5_relu_pointwise_kernel[(triton.cdiv(width, bw), triton.cdiv(height, tile_h), batch)](
        value,
        dw_weight,
        dw_bias,
        pw_weight,
        pw_bias,
        out,
        HEIGHT=height,
        WIDTH=width,
        CHANNELS=channels,
        BW=bw,
        TILE_H=tile_h,
        num_warps=num_warps,
    )
    return out.permute(0, 3, 1, 2)


@triton.jit
def _dw5x5_relu_mx_pointwise_kernel(
    input_ptr,
    dw_weight_ptr,
    dw_bias_ptr,
    pw_weight_ptr,
    pw_scale_ptr,
    pw_bias_ptr,
    output_ptr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    BW: tl.constexpr,
    TILE_H: tl.constexpr,
):
    """Whole MXFP8 refiner block in one kernel: depthwise 5x5 + bias + ReLU (row rolling),
    dynamic per-32-channel MXFP8 quantization, block-scaled ``tl.dot_scaled`` pointwise GEMM
    against the FP8 weight, bias, BF16 channels-last store. The quantized activations never
    leave registers. Quantization math is identical to ``_depthwise5x5_relu_mxfp8_kernel``.
    """
    GROUPS: tl.constexpr = CHANNELS // 32
    x = tl.program_id(0) * BW + tl.arange(0, BW)
    y_start = tl.program_id(1) * TILE_H
    batch = tl.program_id(2)
    x_ok = x < WIDTH
    channel = tl.arange(0, CHANNELS)
    bias = tl.load(dw_bias_ptr + channel).to(tl.float32)
    # Pointwise weight ([out, in] in memory) as the [K=in, N=out] operand, plus its [N, K/32] scales.
    pw_weight = tl.load(pw_weight_ptr + channel[None, :] * CHANNELS + channel[:, None])
    pw_scale = tl.load(pw_scale_ptr + channel[:, None] * GROUPS + tl.arange(0, GROUPS)[None, :])
    pw_bias = tl.load(pw_bias_ptr + channel).to(tl.float32)
    acc0 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc1 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc2 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc3 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc4 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    for step in range(TILE_H + 4):
        input_y = y_start - 2 + step
        row_ok = (input_y >= 0) & (input_y < HEIGHT)
        row_base = (batch * HEIGHT + input_y) * WIDTH
        for kx in tl.static_range(5):
            values = _load_tap_row(input_ptr, row_base, x, row_ok, x_ok, channel, CHANNELS, WIDTH, kx)
            acc4 += values * tl.load(dw_weight_ptr + (0 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc3 += values * tl.load(dw_weight_ptr + (1 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc2 += values * tl.load(dw_weight_ptr + (2 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc1 += values * tl.load(dw_weight_ptr + (3 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc0 += values * tl.load(dw_weight_ptr + (4 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
        out_y = input_y - 2
        if (step >= 4) & (out_y < HEIGHT):
            activated = tl.maximum(acc0 + bias[None, :], 0.0)
            blocks = tl.reshape(activated, (BW, GROUPS, 32))
            amax = tl.max(tl.abs(blocks), axis=2)
            exponent = tl.ceil(tl.log2(amax / 448.0))
            exponent = tl.maximum(tl.minimum(exponent, 127.0), -127.0)
            exponent = tl.where(amax == 0.0, -127.0, exponent)
            quantized = tl.maximum(tl.minimum(blocks * tl.exp2(-exponent)[:, :, None], 448.0), -448.0)
            product = tl.dot_scaled(
                tl.reshape(quantized, (BW, CHANNELS)).to(tl.float8e4nv),
                (exponent + 127.0).to(tl.uint8),
                "e4m3",
                pw_weight,
                pw_scale,
                "e4m3",
            )
            out = product + pw_bias[None, :]
            row = (batch * HEIGHT + out_y) * WIDTH + x
            tl.store(output_ptr + row[:, None] * CHANNELS + channel[None, :], out.to(tl.bfloat16), mask=x_ok[:, None])
        acc0 = acc1
        acc1 = acc2
        acc2 = acc3
        acc3 = acc4
        acc4 = tl.zeros((BW, CHANNELS), dtype=tl.float32)


def depthwise5x5_relu_pointwise_mxfp8(
    value: Tensor,
    dw_weight: Tensor,
    dw_bias: Tensor,
    pw_weight: Tensor,
    pw_scale_u8: Tensor,
    pw_bias: Tensor,
    *,
    bw: int = 32,
    tile_h: int = 32,
    num_warps: int = 8,
) -> Tensor:
    batch, channels, height, width = value.shape
    out = torch.empty((batch, height, width, channels), device=value.device, dtype=torch.bfloat16)
    _dw5x5_relu_mx_pointwise_kernel[(triton.cdiv(width, bw), triton.cdiv(height, tile_h), batch)](
        value,
        dw_weight,
        dw_bias,
        pw_weight,
        pw_scale_u8,
        pw_bias,
        out,
        HEIGHT=height,
        WIDTH=width,
        CHANNELS=channels,
        BW=bw,
        TILE_H=tile_h,
        num_warps=num_warps,
    )
    return out.permute(0, 3, 1, 2)


@triton.jit
def _dw5x5_relu_int8_pointwise_kernel(
    input_ptr,
    dw_weight_ptr,
    dw_bias_ptr,
    pw_weight_ptr,
    pw_wscale_ptr,
    pw_bias_ptr,
    scale_ptr,
    output_ptr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    BW: tl.constexpr,
    TILE_H: tl.constexpr,
):
    """Whole INT8 refiner block in one kernel: depthwise 5x5 + bias + ReLU (row rolling), BF16 rounding,
    static per-tensor INT8 quantization, INT8 ``tl.dot`` against the per-channel INT8 pointwise weight
    (INT32 accumulation), dequantizing epilogue + bias, BF16 channels-last store."""
    x = tl.program_id(0) * BW + tl.arange(0, BW)
    y_start = tl.program_id(1) * TILE_H
    batch = tl.program_id(2)
    x_ok = x < WIDTH
    channel = tl.arange(0, CHANNELS)
    bias = tl.load(dw_bias_ptr + channel).to(tl.float32)
    pw_weight = tl.load(pw_weight_ptr + channel[None, :] * CHANNELS + channel[:, None])  # [K=in, N=out]
    inv_scale = 1.0 / tl.load(scale_ptr)
    out_scale = tl.load(scale_ptr) * tl.load(pw_wscale_ptr + channel)
    pw_bias = tl.load(pw_bias_ptr + channel).to(tl.float32)
    acc0 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc1 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc2 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc3 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    acc4 = tl.zeros((BW, CHANNELS), dtype=tl.float32)
    for step in range(TILE_H + 4):
        input_y = y_start - 2 + step
        row_ok = (input_y >= 0) & (input_y < HEIGHT)
        row_base = (batch * HEIGHT + input_y) * WIDTH
        for kx in tl.static_range(5):
            values = _load_tap_row(input_ptr, row_base, x, row_ok, x_ok, channel, CHANNELS, WIDTH, kx)
            acc4 += values * tl.load(dw_weight_ptr + (0 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc3 += values * tl.load(dw_weight_ptr + (1 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc2 += values * tl.load(dw_weight_ptr + (2 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc1 += values * tl.load(dw_weight_ptr + (3 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
            acc0 += values * tl.load(dw_weight_ptr + (4 * 5 + kx) * CHANNELS + channel).to(tl.float32)[None, :]
        out_y = input_y - 2
        if (step >= 4) & (out_y < HEIGHT):
            activated = tl.maximum(acc0 + bias[None, :], 0.0).to(tl.bfloat16).to(tl.float32)
            scaled = libdevice.rint(activated * inv_scale)
            quantized = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
            product = tl.dot(quantized, pw_weight, out_dtype=tl.int32)
            out = product.to(tl.float32) * out_scale[None, :] + pw_bias[None, :]
            row = (batch * HEIGHT + out_y) * WIDTH + x
            tl.store(output_ptr + row[:, None] * CHANNELS + channel[None, :], out.to(tl.bfloat16), mask=x_ok[:, None])
        acc0 = acc1
        acc1 = acc2
        acc2 = acc3
        acc3 = acc4
        acc4 = tl.zeros((BW, CHANNELS), dtype=tl.float32)


def depthwise5x5_relu_pointwise_int8(
    value: Tensor,
    dw_weight: Tensor,
    dw_bias: Tensor,
    pw_weight: Tensor,
    pw_wscale: Tensor,
    pw_bias: Tensor,
    scale: Tensor,
    *,
    bw: int = 32,
    tile_h: int = 32,
    num_warps: int = 8,
) -> Tensor:
    batch, channels, height, width = value.shape
    out = torch.empty((batch, height, width, channels), device=value.device, dtype=torch.bfloat16)
    _dw5x5_relu_int8_pointwise_kernel[(triton.cdiv(width, bw), triton.cdiv(height, tile_h), batch)](
        value,
        dw_weight,
        dw_bias,
        pw_weight,
        pw_wscale,
        pw_bias,
        scale,
        out,
        HEIGHT=height,
        WIDTH=width,
        CHANNELS=channels,
        BW=bw,
        TILE_H=tile_h,
        num_warps=num_warps,
    )
    return out.permute(0, 3, 1, 2)

# -- registrations ---------------------------------------------------------------------------------------

def _depthwise_eligible(value: Tensor, weight: Tensor, bias: Tensor, *_) -> bool:
    return (
        value.ndim == 4
        and value.shape[1] in (32, 128, 512)
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(weight.shape) == (25, value.shape[1])
    )


ops.depthwise5x5_relu_mxfp8.register(  # an E4M3 producer: see ``backends/triton/quantize.py``
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3], when=_depthwise_eligible
)(depthwise5x5_relu_mxfp8)
ops.depthwise5x5_relu_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_depthwise_eligible
)(depthwise5x5_relu_int8)


def _dw_pw_eligible(value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor) -> bool:
    channels = value.shape[1] if value.ndim == 4 else 0
    return (
        channels in DEPTHWISE5X5_RELU_POINTWISE_CHANNELS
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(dw_weight.shape) == (25, channels)
        and tuple(pw_weight.shape) == (channels, channels)
        and pw_weight.dtype == torch.bfloat16
        and pw_weight.is_contiguous()
        and (value.numel() + 128 * channels) < 2**31
    )


ops.depthwise5x5_relu_pointwise.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_dw_pw_eligible
)(depthwise5x5_relu_pointwise)


def _mxfp8_block_eligible(value, dw_weight, dw_bias, pw_weight, pw_scale_swizzled, pw_scale_u8, pw_bias) -> bool:
    return (
        value.ndim == 4
        and value.shape[1] == 128
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(dw_weight.shape) == (25, 128)
        and tuple(pw_weight.shape) == (128, 128)
        and pw_weight.dtype == torch.float8_e4m3fn
        and pw_scale_u8.dtype == torch.uint8
        and tuple(pw_scale_u8.shape) == (128, 4)
    )


# sm_100+ only (``tl.dot_scaled``). Without block-scaled MMA the composite reference (``depthwise5x5_relu_mxfp8`` ->
# the Triton ``mxfp8_scaled_mm``) is the fast path: a fused kernel with one E4M3 MMA per scale block was bit-identical
# to it but slower on an L20 (2x128x400x400: 0.567 ms against 0.425 ms; 2x128x256x256: 0.214 against 0.166 ms).
@ops.depthwise5x5_relu_pointwise_mxfp8.register(
    "triton_fused", device=DeviceSpec("cuda", min_capability=(10, 0)), requires=[Feature.TRITON],
    when=_mxfp8_block_eligible,
)
def _depthwise5x5_relu_pointwise_mxfp8_impl(value, dw_weight, dw_bias, pw_weight, pw_scale_swizzled, pw_scale_u8, pw_bias):
    return depthwise5x5_relu_pointwise_mxfp8(value, dw_weight, dw_bias, pw_weight, pw_scale_u8, pw_bias)


def _int8_block_eligible(value, dw_weight, dw_bias, pw_weight, pw_wscale, pw_bias, scale) -> bool:
    return (
        value.ndim == 4
        and value.shape[1] == 128
        and value.is_contiguous(memory_format=torch.channels_last)
        and tuple(dw_weight.shape) == (25, 128)
        and tuple(pw_weight.shape) == (128, 128)
        and pw_weight.dtype == torch.int8
    )


ops.depthwise5x5_relu_pointwise_int8.register(
    "triton_fused", device=DeviceSpec("cuda", min_capability=(8, 0)), requires=[Feature.TRITON],
    when=_int8_block_eligible,
)(depthwise5x5_relu_pointwise_int8)


# -- im2col gather ------------------------------------------------------------------------------------


@triton.jit
def _im2col_gather_kernel(
    x_ptr, out_ptr,
    H, W, HO, WO,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    C: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """One program per output row: gathers every tap's ``C`` channels straight from ``x`` (zero-padded on read,
    no materialized padding) and writes them contiguously — the whole im2col row in one kernel, one read of each
    input byte per tap, no intermediate tensors. ``BLOCK_C`` is ``C`` rounded up to a power of 2 (``tl.arange``
    needs one); the channel mask covers both the padding taps and this rounding."""
    row = tl.program_id(0).to(tl.int64)
    b = row // (HO * WO)
    rem = row % (HO * WO)
    oy = rem // WO
    ox = rem % WO
    c = tl.arange(0, BLOCK_C).to(tl.int64)
    c_ok = c < C
    out_base = row * (KH * KW * C)
    for ky in tl.static_range(KH):
        iy = oy * SH + ky * DH - PH
        y_ok = (iy >= 0) & (iy < H)
        for kx in tl.static_range(KW):
            ix = ox * SW + kx * DW - PW
            ok = c_ok & y_ok & (ix >= 0) & (ix < W)
            x_off = ((b * H + iy) * W + ix) * C
            vals = tl.load(x_ptr + x_off + c, mask=ok, other=0)
            tap = ky * KW + kx
            tl.store(out_ptr + out_base + tap * C + c, vals, mask=c_ok)


def _im2col_gather_triton(x: Tensor, kh: int, kw: int, sh: int, sw: int, ph: int, pw: int, dh: int, dw: int) -> Tensor:
    b, h, w, c = x.shape
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    raw = x.view(torch.uint8).contiguous()
    out = torch.empty(b * ho * wo, kh * kw * c, dtype=torch.uint8, device=x.device)
    _im2col_gather_kernel[(b * ho * wo,)](
        raw, out, h, w, ho, wo, KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw, DH=dh, DW=dw,
        C=c, BLOCK_C=triton.next_power_of_2(c),
    )
    return out.view(x.dtype)


def _im2col_eligible(x: Tensor, kh: int, kw: int, sh: int, sw: int, ph: int, pw: int, dh: int, dw: int) -> bool:
    return x.ndim == 4 and x.dtype.itemsize == 1 and x.shape[-1] <= 1024


ops.im2col_gather.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_im2col_eligible
)(_im2col_gather_triton)


# Sizes of the implicit-GEMM convolutions: autotuned per power-of-2 bucket of the row count (``ROWS_BUCKET``), and not
# integer-specialized, so a new image size reuses the tuned config and the compiled kernel. Keyed on the exact row count
# and specialized on divisibility, every new image size re-autotuned and recompiled every config: 6-13 s per new size
# for a SuperPoint int8 / fp8 extract on sm_120 (a MegaDepth evaluation has dozens of sizes).
_SIZE_ARGS = ["B", "H", "W", "HO", "WO", "rows", "ROWS_BUCKET", "res_channels", "rs_b", "rs_h", "rs_w", "rs_c"]
# Keep the autotuner's timings in Triton's cache (keyed by kernel source, architecture, tuning key and candidates), so
# only the first process times the candidates: RoMaV2's eight INT8 VGG convs cost about 10 s of autotuning per process
# start on a Jetson AGX Orin (first forward 12.5 -> 2.7 s) and on an L20. ``FLASHIM_AUTOTUNE_CACHE=0`` re-times.
_CACHE_AUTOTUNE = os.environ.get("FLASHIM_AUTOTUNE_CACHE", "1") != "0"


# -- fused INT8 conv2d (implicit GEMM) ----------------------------------------------------------

# (BLOCK_M, BLOCK_N, num_warps, num_stages): every real-shape winner of the former 11-tile list on sm_120 (DeDoDe /
# RIPE VGG, LoFTR, DISK, SuperPoint convs), plus 32x64 for small row counts; x the BLOCK_K choices below.
_INT8_CONV2D_TILES = [
    (32, 64, 4, 3), (64, 64, 4, 3), (64, 128, 4, 3), (128, 64, 4, 3), (128, 128, 4, 3), (128, 128, 8, 2),
]
_INT8_CONV2D_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
    for bm, bn, nw, ns in _INT8_CONV2D_TILES
    for bk in (32, 64, 128)
]


def _int8_conv2d_block_k(configs, named_args, **kwargs):
    """Channel chunk per ``tl.dot``: ``Cin`` rounded up to a power of 2 (>= 32, ``tl.dot``'s minimum) while that is at
    most 64; wider inputs loop over 64- or 128-channel chunks (autotuned) instead of one ``next_pow2(Cin)``-wide dot,
    which padded LoFTR's 196 channels to 256 and kept 512-wide operands in registers. Measured on sm_120 (module
    forward, channels-last input): 512 -> 512 3x3 at 112x150 0.585 -> 0.318 ms, 196 -> 128 at 448x600 1.309 -> 0.866,
    196 -> 196 at 224x300 0.586 -> 0.442, 256 -> 256 at 224x300 0.412 -> 0.346; 64-channel layers unchanged. INT32
    accumulation is exact, so the chunking does not change any output bit.

    When the last 64-wide chunk would be at most half full, 32-wide chunks are candidates too: ncu on an L20 showed
    LoFTR's 196-channel layers spending their MMAs on 256 padded channels. Autotuned there with them: 196 -> 196 at
    448x600 2.08 -> 1.43 ms and at 224x300 0.522 -> 0.371 ms (32-wide chunks picked), 196 -> 128 0.790 -> 0.746."""
    width = max(32, triton.next_power_of_2(kwargs["CIN"]))
    allowed = (width,) if width <= 64 else (64, 128)
    if width > 64 and 0 < kwargs["CIN"] % 64 <= 32:  # e.g. 196: 32-wide chunks cover 224 channels instead of 256
        allowed = (32, 64, 128)
    return [c for c in configs if c.kwargs["BLOCK_K"] in allowed]


@triton.autotune(
    configs=_INT8_CONV2D_CONFIGS, key=["ROWS_BUCKET", "COUT", "CIN", "TAPS"],
    prune_configs_by={"early_config_prune": _int8_conv2d_block_k}, cache_results=_CACHE_AUTOTUNE,
)
@triton.jit(do_not_specialize=_SIZE_ARGS)
def _int8_conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr, out_scale_ptr,
    a_scale_ptr, w_scale_ptr,
    B, H, W, HO, WO, rows, ROWS_BUCKET,
    res_ptr, res_channels, rs_b, rs_h, rs_w, rs_c,
    CIN: tl.constexpr, COUT: tl.constexpr, TAPS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    HAS_BIAS: tl.constexpr, RELU: tl.constexpr, HAS_RESIDUAL: tl.constexpr, QUANT_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Implicit-GEMM INT8 ``nn.Conv2d`` (``groups == 1``): per filter tap, INT32 ``tl.dot``s over ``BLOCK_K``-channel
    chunks gathered straight from ``x_ptr`` ``[B, H, W, CIN]`` with the tap's spatial offset folded into the address
    (padding masked to zero) instead of materializing ``im2col_gather``'s buffer and running a separate GEMM on it.
    The channel mask covers padding and the last partial chunk, so ``CIN`` need not be a multiple of ``BLOCK_K`` (see
    ``_int8_conv2d_block_k``). Epilogue matches ``_int8_gemm_kernel`` (scales, bias, optional ReLU). ``QUANT_OUT``:
    the output is stored as the next layer's INT8 input (``out_scale_ptr``: its activation scale): rounded to BF16,
    then quantized exactly as ``quantize_int8`` does on the BF16 tensor (same bytes as the separate pass).
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(COUT, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    b = offs_m // (HO * WO)
    rem = offs_m % (HO * WO)
    oh = rem // WO
    ow = rem % WO

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < COUT
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for ky in tl.static_range(KH):
        ih = oh * SH + ky * DH - PH
        ih_ok = (ih >= 0) & (ih < H)
        for kx in tl.static_range(KW):
            iw = ow * SW + kx * DW - PW
            iw_ok = (iw >= 0) & (iw < W)
            row_ok = mask_m & ih_ok & iw_ok
            pixel = ((b * H + ih) * W + iw).to(tl.int64) * CIN
            tap = ky * KW + kx
            for k0 in tl.static_range(0, CIN, BLOCK_K):
                offs_c = k0 + offs_k
                c_ok = offs_c < CIN
                a = tl.load(x_ptr + pixel[:, None] + offs_c[None, :], mask=row_ok[:, None] & c_ok[None, :], other=0)
                b_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * (TAPS * CIN) + tap * CIN + offs_c[:, None]
                wt = tl.load(b_ptrs, mask=mask_n[None, :] & c_ok[:, None], other=0)
                accumulator = tl.dot(a, wt, accumulator, out_dtype=tl.int32)

    scale = tl.load(a_scale_ptr) * tl.load(w_scale_ptr + offs_n, mask=mask_n, other=0.0)
    if HAS_BIAS:  # explicit FMA: the compiler contracted `acc * scale + bias` for most tile configs but not all,
        # so the output bits depended on which config autotuning picked; one rounding for every config instead
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = tl.fma(accumulator.to(tl.float32), scale[None, :], bias[None, :])
    else:
        out = accumulator.to(tl.float32) * scale[None, :]
    if HAS_RESIDUAL:  # added in FP32 before the single BF16 rounding
        res_ptrs = (res_ptr + b[:, None].to(tl.int64) * rs_b + oh[:, None].to(tl.int64) * rs_h
                    + ow[:, None].to(tl.int64) * rs_w + offs_n[None, :].to(tl.int64) * rs_c)
        res_ok = mask_m[:, None] & (offs_n < res_channels)[None, :]
        out += tl.load(res_ptrs, mask=res_ok, other=0.0).to(tl.float32)
    if RELU:
        out = tl.maximum(out, 0.0)
    out_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * COUT + offs_n[None, :]
    if QUANT_OUT:
        scaled = libdevice.rint(out.to(tl.bfloat16).to(tl.float32) * (1.0 / tl.load(out_scale_ptr)))
        quantized = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
        tl.store(out_ptrs, quantized, mask=mask_m[:, None] & mask_n[None, :])
    else:
        tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _residual_args(residual: Tensor | None, out: Tensor) -> tuple:
    """``(pointer, channels, 4 strides)`` of an optional ``[B, Ho, Wo, C]`` residual (any strides; e.g. the
    channels-last view of an NCHW tensor)."""
    if residual is None:
        return (out, 0, 0, 0, 0, 0)
    return (residual, residual.shape[-1], *residual.stride())


def _int8_conv2d_triton(
    x, weight, a_scale, w_scale, bias, kh, kw, sh, sw, ph, pw, dh, dw, *, relu=False, residual=None, out_scale=None
) -> Tensor:
    b, h, w, cin = x.shape
    cout = weight.shape[0]
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    quantize = out_scale is not None
    out = torch.empty(b, ho, wo, cout, device=x.device, dtype=torch.int8 if quantize else torch.bfloat16)
    rows = b * ho * wo
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(cout, meta["BLOCK_N"]),)
    _int8_conv2d_kernel[grid](
        x, weight, bias, out, out_scale if quantize else a_scale, a_scale, w_scale,
        b, h, w, ho, wo, rows, triton.next_power_of_2(rows), *_residual_args(residual, out),
        CIN=cin, COUT=cout, TAPS=kh * kw, KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw, DH=dh, DW=dw,
        HAS_BIAS=bias is not None, RELU=relu, HAS_RESIDUAL=residual is not None, QUANT_OUT=quantize,
    )
    return out


def _int8_conv2d_eligible(x, weight, a_scale, w_scale, bias, kh, kw, sh, sw, ph, pw, dh, dw, **_) -> bool:
    return (
        x.ndim == 4
        and x.dtype == torch.int8
        and x.is_contiguous()
        and weight.shape == (weight.shape[0], kh * kw * x.shape[-1])
        and kh * kw <= 49  # keeps the unrolled tap loop's compile time reasonable
    )


ops.int8_conv2d.register(
    "triton", device=DeviceSpec("cuda", min_capability=(7, 5)), requires=[Feature.TRITON],
    when=_int8_conv2d_eligible,
)(_int8_conv2d_triton)


# -- fused static MXFP8 conv2d (implicit GEMM) --------------------------------------------------

_MXFP8_CONV2D_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bm, bn, nw, ns in [
        (32, 32, 4, 3), (32, 64, 4, 3), (64, 32, 4, 3),
        (64, 64, 4, 3), (64, 64, 8, 3), (64, 128, 4, 3),
        (128, 64, 4, 3), (128, 128, 4, 3), (128, 128, 8, 2),
    ]
]


def _mxfp8_conv2d_tiles(configs, named_args, **kwargs):
    """Hopper: 64+-row tiles only. Triton lowers them to wgmma and 32-row ones to mma.sync there, and the two E4M3 MMAs
    round differently (``backends/triton/gemm._is_sm90``), so the output bits would depend on which tile the autotuner
    happened to time fastest. The 64- and 128-row tiles won every shape measured on an H20 (64 -> 64 and 128 -> 128
    3x3, 256 -> 256, 512 -> 512, 128 -> 128 5x5)."""
    if platform_for(named_args["x_ptr"].device).capability == (9, 0):
        return [c for c in configs if c.kwargs["BLOCK_M"] >= 64]
    return configs


@triton.autotune(
    configs=_MXFP8_CONV2D_CONFIGS, key=["ROWS_BUCKET", "COUT", "CIN", "TAPS"],
    prune_configs_by={"early_config_prune": _mxfp8_conv2d_tiles}, cache_results=_CACHE_AUTOTUNE,
)
@triton.jit(do_not_specialize=_SIZE_ARGS)
def _mxfp8_conv2d_kernel(
    x_ptr, w_ptr, wexp_ptr, cexp_ptr, bias_ptr, out_ptr, oexp_ptr,
    B, H, W, HO, WO, rows, ROWS_BUCKET,
    res_ptr, res_channels, rs_b, rs_h, rs_w, rs_c,
    CIN: tl.constexpr, COUT: tl.constexpr, TAPS: tl.constexpr, CIN_BLOCKS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    HAS_BIAS: tl.constexpr, RELU: tl.constexpr, HAS_RESIDUAL: tl.constexpr, QUANT_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Implicit-GEMM static MXFP8 ``nn.Conv2d`` (``groups == 1``): one 32-channel-block E4M3 ``tl.dot`` per
    (tap, channel block), gathered straight from ``x_ptr`` ``[B, H, W, CIN]`` (``CIN`` already a multiple of 32),
    instead of materializing ``im2col_gather``'s buffer and running ``mxfp8_linear`` on it. The activation's
    per-channel-block exponent (``cexp_ptr``, ``[CIN / 32]``) is the *same* for every tap -- a tap only changes
    which weight reads a pixel, not the pixel's own scale -- so it's read once per (tap, block) instead of the
    unfused path's tap-duplicated copy; the weight's own exponent (``wexp_ptr``, ``[COUT, TAPS * CIN / 32]``) does
    vary per tap (it's a different weight tap, not a different view of the same activation). FP32 accumulation,
    scaled per block as ``sum(a_bits * b_bits) * 2**a_exp * 2**b_exp`` (matches ``_fp8_gemm_kernel``'s epilogue).

    The (tap, channel block) pairs are one run-time loop, so Triton pipelines the operand loads (``num_stages``)
    instead of emitting ``TAPS * CIN_BLOCKS`` unrolled, unpipelined dots. Same summation order: bit-identical to the
    unrolled kernel and to ``im2col_gather`` + ``_fp8_gemm_kernel``. Measured on an L20 (sm_89; sm_90+ compiled only)
    against the unrolled kernel: 64 -> 64 3x3 at 896x1200 0.623 -> 0.470 ms, 128 -> 128 3x3 at 448x600 0.946 -> 0.429,
    128 -> 256 3x3 at 112x150 0.123 -> 0.062, 512 -> 512 1x1 at 112x150 0.082 -> 0.056; and against im2col + GEMM
    on the shapes the unrolled kernel lost: 256 -> 256 3x3 at 224x300 0.774 -> 0.416 ms (72 dots), 512 -> 512 3x3
    at 112x150 0.576 -> 0.429 (144), 224 -> 224 3x3 at 448x600 5.28 -> 1.43 (63), 128 -> 128 5x5 at 448x600 3.69
    -> 1.13 (100): see ``fused_mxfp8_conv2d_worthwhile``.

    ``QUANT_OUT``: the output is stored as the next layer's static MXFP8 input (``oexp_ptr``: its ``[COUT / 32]``
    block exponents) instead of BF16: rounded to BF16 first, then scaled, saturated and cast exactly as
    ``_quantize_static_kernel`` does on the BF16 tensor, so the bytes are the ones the separate quantize pass writes.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(COUT, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    b = offs_m // (HO * WO)
    rem = offs_m % (HO * WO)
    oh = rem // WO
    ow = rem % WO

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < COUT
    offs_block = tl.arange(0, 32)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(TAPS * CIN_BLOCKS):  # one run-time loop over (tap, channel block): see the docstring
        tap = kb // CIN_BLOCKS
        kt = kb % CIN_BLOCKS
        ih = oh * SH + (tap // KW) * DH - PH
        iw = ow * SW + (tap % KW) * DW - PW
        row_ok = mask_m & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        a_ptrs = x_ptr + ((b * H + ih) * W + iw)[:, None].to(tl.int64) * CIN + (kt * 32 + offs_block)[None, :]
        a = tl.load(a_ptrs, mask=row_ok[:, None], other=0.0)
        a_scale = tl.exp2(tl.load(cexp_ptr + kt).to(tl.float32))

        w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * (TAPS * CIN) + kb * 32 + offs_block[None, :]
        wt = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0)
        w_scale = tl.exp2(tl.load(wexp_ptr + offs_n * (TAPS * CIN_BLOCKS) + kb, mask=mask_n, other=-127.0).to(tl.float32))

        accumulator += tl.dot(a, tl.trans(wt)) * (a_scale * w_scale[None, :])

    out = accumulator
    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    if HAS_RESIDUAL:  # added in FP32 before the single BF16 rounding
        res_ptrs = (res_ptr + b[:, None].to(tl.int64) * rs_b + oh[:, None].to(tl.int64) * rs_h
                    + ow[:, None].to(tl.int64) * rs_w + offs_n[None, :].to(tl.int64) * rs_c)
        res_ok = mask_m[:, None] & (offs_n < res_channels)[None, :]
        out += tl.load(res_ptrs, mask=res_ok, other=0.0).to(tl.float32)
    if RELU:
        out = tl.maximum(out, 0.0)
    out_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * COUT + offs_n[None, :]
    if QUANT_OUT:
        dense = out.to(tl.bfloat16).to(tl.float32)
        exponent = tl.load(oexp_ptr + offs_n // 32, mask=mask_n, other=0).to(tl.float32)
        scaled = dense * tl.exp2(-exponent)[None, :]
        quantized = to_e4m3(tl.maximum(tl.minimum(scaled, 448.0), -448.0))
        tl.store(out_ptrs, quantized, mask=mask_m[:, None] & mask_n[None, :])
    else:
        tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _mxfp8_conv2d_triton(
    x, channel_exponent, weight, weight_exponent, bias, kh, kw, sh, sw, ph, pw, dh, dw, *, relu=False, residual=None,
    out_exponent=None,
) -> Tensor:
    b, h, w, cin = x.shape
    cout = weight.shape[0]
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    quantize = out_exponent is not None
    out = torch.empty(b, ho, wo, cout, device=x.device, dtype=torch.float8_e4m3fn if quantize else torch.bfloat16)
    rows = b * ho * wo
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(cout, meta["BLOCK_N"]),)
    _mxfp8_conv2d_kernel[grid](
        x, weight, weight_exponent, channel_exponent, bias, out, out_exponent if quantize else channel_exponent,
        b, h, w, ho, wo, rows, triton.next_power_of_2(rows), *_residual_args(residual, out),
        CIN=cin, COUT=cout, TAPS=kh * kw, CIN_BLOCKS=cin // 32, KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw, DH=dh, DW=dw,
        HAS_BIAS=bias is not None, RELU=relu, HAS_RESIDUAL=residual is not None, QUANT_OUT=quantize,
    )
    return out


def _mxfp8_conv2d_eligible(
    x, channel_exponent, weight, weight_exponent, bias, kh, kw, sh, sw, ph, pw, dh, dw, **_
) -> bool:
    return (
        x.ndim == 4
        and x.dtype == torch.float8_e4m3fn
        and x.is_contiguous()
        and x.shape[-1] % 32 == 0
        and weight.shape == (weight.shape[0], kh * kw * x.shape[-1])
        and fused_mxfp8_conv2d_worthwhile(kh * kw, x.shape[-1] // 32, platform_for(x.device))
    )


ops.mxfp8_conv2d.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3],
    when=_mxfp8_conv2d_eligible,
)(_mxfp8_conv2d_triton)


# -- fused static MXFP8 conv2d with block-scaled MMA (tl.dot_scaled) ------------------------------------

# (BLOCK_M, BLOCK_N, num_warps, num_stages) x BLOCK_K in {32, 64, 128} (dividing CIN): the tiles that won at the real
# shapes on sm_120 among 8 candidates (VGG 64-512 ch, LoFTR 128-224, DISK 5x5 96), plus 64x64 for small row counts.
_MXFP8_CONV2D_SCALED_TILES = [(64, 64, 4, 3), (128, 64, 4, 3), (128, 128, 4, 3), (256, 128, 8, 2)]
_MXFP8_CONV2D_SCALED_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
    for bm, bn, nw, ns in _MXFP8_CONV2D_SCALED_TILES
    for bk in (32, 64, 128)
]


def _mxfp8_conv2d_scaled_block_k(configs, named_args, **kwargs):
    """``BLOCK_K`` must divide ``CIN`` (a multiple of 32): every chunk is whole 32-channel scale blocks."""
    return [c for c in configs if kwargs["CIN"] % c.kwargs["BLOCK_K"] == 0]


@triton.autotune(
    configs=_MXFP8_CONV2D_SCALED_CONFIGS, key=["ROWS_BUCKET", "COUT", "CIN", "TAPS"],
    prune_configs_by={"early_config_prune": _mxfp8_conv2d_scaled_block_k}, cache_results=_CACHE_AUTOTUNE,
)
@triton.jit(do_not_specialize=_SIZE_ARGS[1:])
def _mxfp8_conv2d_scaled_kernel(
    x_ptr, w_ptr, wexp_ptr, cexp_ptr, bias_ptr, out_ptr,
    H, W, HO, WO, rows, ROWS_BUCKET,
    res_ptr, res_channels, rs_b, rs_h, rs_w, rs_c,
    CIN: tl.constexpr, COUT: tl.constexpr, TAPS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    HAS_BIAS: tl.constexpr, RELU: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Implicit-GEMM static MXFP8 ``nn.Conv2d`` on block-scaled tensor cores: per filter tap, one ``tl.dot_scaled``
    per ``BLOCK_K``-channel chunk (Triton 3.6 lowers it to ``mma.sync...kind::mxf8f6f4.block_scale`` on sm_120 /
    tcgen05 on sm_100), so the E8M0 scales are applied inside the MMA instead of a per-32-channel FP32 rescale of the
    accumulator (``_mxfp8_conv2d_kernel``). Activation scales are the static per-channel-block exponents (the same for
    every pixel and tap, broadcast over rows); weight scales are per (output channel, tap, 32 inputs). Gathered
    straight from ``x_ptr`` ``[B, H, W, CIN]`` with the tap offset in the address (padding masked to zero)."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(COUT, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    b = offs_m // (HO * WO)
    rem = offs_m % (HO * WO)
    oh = rem // WO
    ow = rem % WO
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < COUT
    offs_k = tl.arange(0, BLOCK_K)
    GROUPS: tl.constexpr = BLOCK_K // 32
    offs_g = tl.arange(0, GROUPS)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ky in tl.static_range(KH):
        ih = oh * SH + ky * DH - PH
        ih_ok = (ih >= 0) & (ih < H)
        for kx in tl.static_range(KW):
            iw = ow * SW + kx * DW - PW
            row_ok = mask_m & ih_ok & (iw >= 0) & (iw < W)
            pixel = ((b * H + ih) * W + iw).to(tl.int64) * CIN
            tap = ky * KW + kx
            for k0 in tl.static_range(0, CIN, BLOCK_K):
                a = tl.load(x_ptr + pixel[:, None] + (k0 + offs_k)[None, :], mask=row_ok[:, None], other=0.0)
                a_exp = tl.load(cexp_ptr + k0 // 32 + offs_g).to(tl.int32) + 127
                a_scale = tl.broadcast_to(a_exp[None, :], (BLOCK_M, GROUPS)).to(tl.uint8)
                column = tap * CIN + k0
                wt = tl.load(
                    w_ptr + offs_n[None, :].to(tl.int64) * (TAPS * CIN) + column + offs_k[:, None],
                    mask=mask_n[None, :], other=0.0,
                )
                w_exp = tl.load(
                    wexp_ptr + offs_n[:, None].to(tl.int64) * (TAPS * CIN // 32) + column // 32 + offs_g[None, :],
                    mask=mask_n[:, None], other=0,
                ).to(tl.int32) + 127
                accumulator = tl.dot_scaled(a, a_scale, "e4m3", wt, w_exp.to(tl.uint8), "e4m3", accumulator)
    out = accumulator
    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    if HAS_RESIDUAL:  # added in FP32 before the single BF16 rounding
        res_ptrs = (res_ptr + b[:, None].to(tl.int64) * rs_b + oh[:, None].to(tl.int64) * rs_h
                    + ow[:, None].to(tl.int64) * rs_w + offs_n[None, :].to(tl.int64) * rs_c)
        res_ok = mask_m[:, None] & (offs_n < res_channels)[None, :]
        out += tl.load(res_ptrs, mask=res_ok, other=0.0).to(tl.float32)
    if RELU:
        out = tl.maximum(out, 0.0)
    out_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * COUT + offs_n[None, :]
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _mxfp8_conv2d_scaled_triton(
    x, channel_exponent, weight, weight_exponent, bias, kh, kw, sh, sw, ph, pw, dh, dw, *, relu=False, residual=None
) -> Tensor:
    b, h, w, cin = x.shape
    cout = weight.shape[0]
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = torch.empty(b, ho, wo, cout, device=x.device, dtype=torch.bfloat16)
    rows = b * ho * wo
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(cout, meta["BLOCK_N"]),)
    _mxfp8_conv2d_scaled_kernel[grid](
        x, weight, weight_exponent, channel_exponent, bias, out, h, w, ho, wo, rows, triton.next_power_of_2(rows), *_residual_args(residual, out),
        CIN=cin, COUT=cout, TAPS=kh * kw, KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw, DH=dh, DW=dw,
        HAS_BIAS=bias is not None, RELU=relu, HAS_RESIDUAL=residual is not None,
    )
    return out


def _mxfp8_conv2d_scaled_eligible(
    x, channel_exponent, weight, weight_exponent, bias, kh, kw, sh, sw, ph, pw, dh, dw, *, out_exponent=None, **_
) -> bool:
    return (
        out_exponent is None  # a quantized output is the per-32 kernel's (``_mxfp8_conv2d_kernel``'s ``QUANT_OUT``)
        and x.ndim == 4
        and x.dtype == torch.float8_e4m3fn
        and x.is_contiguous()
        and x.shape[-1] % 32 == 0
        and weight.shape == (weight.shape[0], kh * kw * x.shape[-1])
        and weight_exponent.is_contiguous()
        and kh * kw <= 49  # keeps the unrolled tap loop's compile time reasonable
    )


# Block-scaled MMA: every shape (no tap x channel-block gate): measured on sm_120 vs the per-32 kernel / im2col +
# GEMM path it replaces (module forward incl. quantize): VGG 64 ch at 896x1200 0.98 -> 0.63 ms, 128 ch at 448x600
# 1.35 -> 0.47, 512 ch at 112x150 0.43 -> 0.38, LoFTR 196 -> 128 at 448x600 2.72 -> 0.88, DISK 5x5 96 -> 64 2.32 -> 0.50
# (results/kernel_runs/mxfp8_conv2d/2026-10-03_dot_scaled_probe); 130-238 TFLOPS against cuDNN BF16's 48-93.
ops.mxfp8_conv2d.register(
    "triton_scaled",
    device=DeviceSpec("cuda", *BLOCK_SCALED_CONV_CAPABILITY),  # sm_12x only: see ops.conv.BLOCK_SCALED_CONV_CAPABILITY
    requires=[Feature.TRITON, Feature.MXFP8_SCALED_MM],
    priority=1,
    when=_mxfp8_conv2d_scaled_eligible,
)(_mxfp8_conv2d_scaled_triton)


# -- single-output-channel FP32 convolution (stencil) ---------------------------------------------------


@triton.jit
def _conv2d_single_output_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C, H, W, stride_b, stride_c, stride_h, stride_w,
    K: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """One output channel, FP32 accumulation: each program owns a ``BLOCK_H x BLOCK_W`` pixel tile and, per input
    channel, adds its ``K x K`` shifted (masked, L1-resident) tiles times the scalar weights. Reads ``x`` in its own
    dtype and strides (no FP32 / layout copy of the input)."""
    oy = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ox = (tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    base = x_ptr + tl.program_id(2).to(tl.int64) * stride_b
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for c in range(C):
        plane = base + c.to(tl.int64) * stride_c
        for ky in tl.static_range(K):
            iy = oy + (ky - K // 2)
            y_ok = (iy >= 0) & (iy < H)
            for kx in tl.static_range(K):
                ix = ox + (kx - K // 2)
                ok = y_ok & (ix >= 0) & (ix < W)
                value = tl.load(plane + iy * stride_h + ix * stride_w, mask=ok, other=0.0).to(tl.float32)
                acc += value * tl.load(w_ptr + (c * K + ky) * K + kx)
    if HAS_BIAS:
        acc += tl.load(bias_ptr)
    out = out_ptr + tl.program_id(2).to(tl.int64) * H * W + oy * W + ox
    tl.store(out, acc, mask=(oy < H) & (ox < W))


@triton.jit
def _conv1x1_single_output_nhwc_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr, C, pixels,
    HAS_BIAS: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """The 1x1 case on channels-last memory: a pixel's channels are contiguous, so ``[BLOCK_P, BLOCK_C]`` tiles are
    loaded whole and reduced over channels (the stencil kernel would read each channel with a ``C``-element stride)."""
    p = (tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)).to(tl.int64)
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        ok = (p < pixels)[:, None] & (c < C)[None, :]
        value = tl.load(x_ptr + p[:, None] * C + c[None, :], mask=ok, other=0.0).to(tl.float32)
        acc += tl.sum(value * tl.load(w_ptr + c, mask=c < C, other=0.0)[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(bias_ptr)
    tl.store(out_ptr + p, acc, mask=p < pixels)


def _conv2d_single_output_triton(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    batch, channels, height, width = x.shape
    k = weight.shape[-1]
    out = torch.empty(batch, 1, height, width, device=x.device, dtype=torch.float32)
    if k == 1 and x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
        w = weight.float().reshape(-1).contiguous()
        pixels = batch * height * width
        block_p, block_c = 64, min(128, triton.next_power_of_2(channels))
        _conv1x1_single_output_nhwc_kernel[(triton.cdiv(pixels, block_p),)](
            x, w, bias.float() if bias is not None else w, out, channels, pixels,
            HAS_BIAS=bias is not None, BLOCK_P=block_p, BLOCK_C=block_c, num_warps=4,
        )
        return out
    # 8 x 128 tiles, 8 warps: 0.745 ms for DISK's 80 -> 1 5x5 at 896x1200 (sm_120; 16x64 / 4 warps 1.04, 32x128 0.96;
    # cuDNN TF32 plus the FP32 copy of the input it needs: 4.95 ms)
    block_h, block_w = 8, 128
    grid = (triton.cdiv(width, block_w), triton.cdiv(height, block_h), batch)
    w = weight.float().contiguous()
    _conv2d_single_output_kernel[grid](
        x, w, bias.float() if bias is not None else w, out, channels, height, width, *x.stride(),
        K=k, HAS_BIAS=bias is not None, BLOCK_H=block_h, BLOCK_W=block_w, num_warps=8,
    )
    return out


def _conv2d_single_output_eligible(x: Tensor, weight: Tensor, bias: Tensor | None) -> bool:
    return (
        x.ndim == 4
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and weight.shape[0] == 1
        and weight.shape[1] == x.shape[1]
        and weight.shape[-1] == weight.shape[-2]
        and weight.shape[-1] % 2 == 1
    )


ops.conv2d_single_output.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_conv2d_single_output_eligible
)(_conv2d_single_output_triton)
