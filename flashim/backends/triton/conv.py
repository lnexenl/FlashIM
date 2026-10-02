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

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.ops.conv import fused_mxfp8_conv2d_worthwhile
from flashim.platforms import Feature


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
):
    """Depthwise 5x5 + bias + ReLU + dynamic MXFP8 (or, ``INT8``, static per-tensor INT8: ``scale_bits_ptr``
    is then the FP32 scale and the ``float8`` output is ``int8``, activations rounded to BF16 first) over a ``BW x TILE_H`` tile, 32 channels.

    Row rolling: each input row's 5 horizontal shifts are loaded once and accumulated into
    the 5 output rows that use it (5 loads per pixel instead of 25). Per output pixel the
    FP32 accumulation order is ky-major, kx-minor, as in a direct 25-tap loop.
    """
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
                tl.store(output_ptr + row[:, None] * CHANNELS + channel[None, :], quantized, mask=x_ok[:, None])
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


def depthwise5x5_relu_int8(
    value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor, *, bw: int = 16, tile_h: int | None = None,
    num_warps: int = 2,
) -> Tensor:
    """Depthwise 5x5 + bias + ReLU -> INT8 ``[B*H*W, C]`` with one static FP32 ``scale``."""
    batch, channels, height, width = value.shape
    tile_h = tile_h or (32 if height > 64 else 16)
    quantized = torch.empty((batch * height * width, channels), device=value.device, dtype=torch.int8)
    _depthwise5x5_relu_mxfp8_kernel[(triton.cdiv(width, bw), triton.cdiv(height, tile_h), batch * (channels // 32))](
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
        num_warps=num_warps,
    )
    return quantized


def depthwise5x5_relu_mxfp8(
    value: Tensor, weight: Tensor, bias: Tensor, *, bw: int = 16, tile_h: int | None = None, num_warps: int = 2
) -> tuple[Tensor, Tensor]:
    batch, channels, height, width = value.shape
    groups = channels // 32
    scale_tiles = (groups + 3) // 4
    rows = batch * height * width
    padded_rows = triton.cdiv(rows, 128) * 128
    tile_h = tile_h or (32 if height > 64 else 16)
    quantized = torch.empty((rows, channels), device=value.device, dtype=torch.float8_e4m3fn)
    scale_bits = torch.zeros(padded_rows * scale_tiles * 4, device=value.device, dtype=torch.uint8)
    _depthwise5x5_relu_mxfp8_kernel[(triton.cdiv(width, bw), triton.cdiv(height, tile_h), batch * groups)](
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
    num_warps: int = 2,
) -> Tensor:
    batch, channels, height, width = value.shape
    if (batch * height * width + bw) * channels >= 2**31:
        raise ValueError("depthwise5x5_relu_pointwise: tensor too large for int32 offsets")
    tile_h = tile_h or (32 if height > 64 else 16)
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


ops.depthwise5x5_relu_mxfp8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_depthwise_eligible
)(depthwise5x5_relu_mxfp8)
ops.depthwise5x5_relu_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_depthwise_eligible
)(depthwise5x5_relu_int8)


def _dw_pw_eligible(value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor) -> bool:
    channels = value.shape[1] if value.ndim == 4 else 0
    return (
        channels in (16, 32, 64)
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


# -- fused INT8 conv2d (implicit GEMM) ----------------------------------------------------------

_INT8_CONV2D_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bm, bn, nw, ns in [
        (32, 32, 4, 3), (32, 64, 4, 3), (64, 32, 4, 3),
        (64, 64, 4, 3), (64, 64, 8, 3), (64, 128, 4, 3),
        (128, 64, 4, 3), (128, 128, 4, 3), (128, 128, 8, 2),
        (256, 64, 8, 2), (256, 128, 8, 2),
    ]
]


@triton.autotune(configs=_INT8_CONV2D_CONFIGS, key=["rows", "COUT", "CIN", "TAPS"])
@triton.jit
def _int8_conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    a_scale_ptr, w_scale_ptr,
    B, H, W, HO, WO, rows,
    CIN: tl.constexpr, COUT: tl.constexpr, TAPS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Implicit-GEMM INT8 ``nn.Conv2d`` (``groups == 1``): one full-``BLOCK_K`` INT32 ``tl.dot`` per filter
    tap, gathered straight from ``x_ptr`` ``[B, H, W, CIN]`` with the tap's spatial offset folded into the
    address (padding masked to zero) instead of materializing ``im2col_gather``'s buffer and running a
    separate GEMM on it. ``BLOCK_K`` is ``CIN`` rounded up to a power of 2 (>= 32, ``tl.dot``'s minimum);
    the channel mask covers padding and this rounding, so ``CIN`` need not itself be a power of 2 (same
    convention as ``_im2col_gather_kernel``'s ``BLOCK_C``). Epilogue matches ``_int8_gemm_kernel``.
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
    offs_c = tl.arange(0, BLOCK_K)
    c_ok = offs_c < CIN

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for ky in tl.static_range(KH):
        ih = oh * SH + ky * DH - PH
        ih_ok = (ih >= 0) & (ih < H)
        for kx in tl.static_range(KW):
            iw = ow * SW + kx * DW - PW
            iw_ok = (iw >= 0) & (iw < W)
            row_ok = mask_m & ih_ok & iw_ok
            a_ptrs = x_ptr + ((b * H + ih) * W + iw)[:, None].to(tl.int64) * CIN + offs_c[None, :]
            a = tl.load(a_ptrs, mask=row_ok[:, None] & c_ok[None, :], other=0)
            tap = ky * KW + kx
            b_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * (TAPS * CIN) + tap * CIN + offs_c[:, None]
            wt = tl.load(b_ptrs, mask=mask_n[None, :] & c_ok[:, None], other=0)
            accumulator = tl.dot(a, wt, accumulator, out_dtype=tl.int32)

    scale = tl.load(a_scale_ptr) * tl.load(w_scale_ptr + offs_n, mask=mask_n, other=0.0)
    out = accumulator.to(tl.float32) * scale[None, :]
    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    out_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * COUT + offs_n[None, :]
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _int8_conv2d_triton(x, weight, a_scale, w_scale, bias, kh, kw, sh, sw, ph, pw, dh, dw) -> Tensor:
    b, h, w, cin = x.shape
    cout = weight.shape[0]
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = torch.empty(b, ho, wo, cout, device=x.device, dtype=torch.bfloat16)
    rows = b * ho * wo
    block_k = max(32, triton.next_power_of_2(cin))
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(cout, meta["BLOCK_N"]),)
    _int8_conv2d_kernel[grid](
        x, weight, bias, out, a_scale, w_scale, b, h, w, ho, wo, rows,
        CIN=cin, COUT=cout, TAPS=kh * kw, KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw, DH=dh, DW=dw,
        HAS_BIAS=bias is not None, BLOCK_K=block_k,
    )
    return out


def _int8_conv2d_eligible(x, weight, a_scale, w_scale, bias, kh, kw, sh, sw, ph, pw, dh, dw) -> bool:
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


@triton.autotune(configs=_MXFP8_CONV2D_CONFIGS, key=["rows", "COUT", "CIN", "TAPS"])
@triton.jit
def _mxfp8_conv2d_kernel(
    x_ptr, w_ptr, wexp_ptr, cexp_ptr, bias_ptr, out_ptr,
    B, H, W, HO, WO, rows,
    CIN: tl.constexpr, COUT: tl.constexpr, TAPS: tl.constexpr, CIN_BLOCKS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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
    for ky in tl.static_range(KH):
        ih = oh * SH + ky * DH - PH
        ih_ok = (ih >= 0) & (ih < H)
        for kx in tl.static_range(KW):
            iw = ow * SW + kx * DW - PW
            iw_ok = (iw >= 0) & (iw < W)
            row_ok = mask_m & ih_ok & iw_ok
            tap = ky * KW + kx
            for kt in tl.static_range(CIN_BLOCKS):
                offs_c = kt * 32 + offs_block
                a_ptrs = x_ptr + ((b * H + ih) * W + iw)[:, None].to(tl.int64) * CIN + offs_c[None, :]
                a = tl.load(a_ptrs, mask=row_ok[:, None], other=0.0)
                a_scale = tl.exp2(tl.load(cexp_ptr + kt).to(tl.float32))

                kb = tap * CIN_BLOCKS + kt
                w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * (TAPS * CIN) + kb * 32 + offs_block[None, :]
                wt = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0)
                w_scale = tl.exp2(tl.load(wexp_ptr + offs_n * (TAPS * CIN_BLOCKS) + kb, mask=mask_n, other=-127.0).to(tl.float32))

                accumulator += tl.dot(a, tl.trans(wt)) * (a_scale * w_scale[None, :])

    out = accumulator
    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    out_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * COUT + offs_n[None, :]
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _mxfp8_conv2d_triton(x, channel_exponent, weight, weight_exponent, bias, kh, kw, sh, sw, ph, pw, dh, dw) -> Tensor:
    b, h, w, cin = x.shape
    cout = weight.shape[0]
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = torch.empty(b, ho, wo, cout, device=x.device, dtype=torch.bfloat16)
    rows = b * ho * wo
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(cout, meta["BLOCK_N"]),)
    _mxfp8_conv2d_kernel[grid](
        x, weight, weight_exponent, channel_exponent, bias, out, b, h, w, ho, wo, rows,
        CIN=cin, COUT=cout, TAPS=kh * kw, CIN_BLOCKS=cin // 32, KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw, DH=dh, DW=dw,
        HAS_BIAS=bias is not None,
    )
    return out


def _mxfp8_conv2d_eligible(x, channel_exponent, weight, weight_exponent, bias, kh, kw, sh, sw, ph, pw, dh, dw) -> bool:
    return (
        x.ndim == 4
        and x.dtype == torch.float8_e4m3fn
        and x.is_contiguous()
        and x.shape[-1] % 32 == 0
        and weight.shape == (weight.shape[0], kh * kw * x.shape[-1])
        and fused_mxfp8_conv2d_worthwhile(kh * kw, x.shape[-1] // 32)
    )


ops.mxfp8_conv2d.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3],
    when=_mxfp8_conv2d_eligible,
)(_mxfp8_conv2d_triton)
