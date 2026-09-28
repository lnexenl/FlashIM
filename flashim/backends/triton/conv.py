"""Depthwise 5x5 conv-block kernels (ported from RoMaV2 ``refiner_fused_mxfp8.py``).

- ``depthwise5x5_relu_mxfp8``: depthwise + bias + ReLU -> dynamic per-(row, 32-channel)
  MXFP8, scales written directly in the ``SWIZZLE_32_4_4`` layout ``F.scaled_mm`` expects;
- ``depthwise5x5_relu_int8``: the same producer with a static per-tensor INT8 output;
- ``depthwise5x5_relu_pointwise``: BF16 depthwise + ReLU + pointwise 1x1 in one pass;
- ``depthwise5x5_relu_pointwise_mxfp8``: the MXFP8 variant with the pointwise GEMM fused.
"""

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
