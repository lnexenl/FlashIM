"""MXFP8 producer kernels (ported from RoMaV2 ``fast_fp8_fusion.py``,
``refiner_fused_mxfp8.py`` and ``full_fp8.py``).

Numerical contracts are kept byte-identical to the RoMaV2 kernels:
- norm/activation math in FP32, rounded to the dense dtype before quantization;
- static quantization saturates to +-448 (E4M3);
- the depthwise producer computes a dynamic per-(row, 32-channel) E8M0 scale and
  writes it directly in the ``SWIZZLE_32_4_4`` layout ``F.scaled_mm`` expects.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim.quant.formats.mx import padded_channels
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight

_ROUND_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


@triton.jit
def _round_to_dense(value, ROUND: tl.constexpr):
    if ROUND == 1:
        return value.to(tl.bfloat16).to(tl.float32)
    if ROUND == 2:
        return value.to(tl.float16).to(tl.float32)
    return value.to(tl.float32)


@triton.jit
def _store_quantized(out_ptr, row, cols, activated, scale_ptr, padded_channels, valid=True):
    store = (cols < padded_channels) & valid
    exponent = tl.load(scale_ptr + cols // 32, mask=store, other=0).to(tl.float32)
    scaled = activated * tl.exp2(-exponent)
    quantized = tl.maximum(tl.minimum(scaled, 448.0), -448.0).to(tl.float8e4nv)
    tl.store(out_ptr + row * padded_channels + cols, quantized, mask=store)


@triton.jit
def _gelu_quantize_kernel(
    value_ptr,
    scale_ptr,
    out_ptr,
    channels: tl.constexpr,
    padded_channels: tl.constexpr,
    elements,
    BLOCK: tl.constexpr,
    TANH: tl.constexpr,
    ROUND: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // padded_channels
    cols = offsets % padded_channels
    valid = offsets < elements
    value = tl.load(value_ptr + row * channels + cols, mask=valid & (cols < channels), other=0.0).to(
        tl.float32
    )
    if TANH:
        inner = 0.7978845608028654 * (value + 0.044715 * value * value * value)
        tanh = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
        activated = 0.5 * value * (1.0 + tanh)
    else:
        activated = 0.5 * value * (1.0 + tl.math.erf(value * 0.7071067811865476))
    activated = _round_to_dense(activated, ROUND)
    _store_quantized(out_ptr, row, cols, activated, scale_ptr, padded_channels, valid)


@triton.jit
def _layernorm_quantize_kernel(
    value_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    out_ptr,
    channels,
    padded_channels,
    eps,
    BLOCK_C: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROUND: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    value_base = row * channels
    partial = tl.zeros([BLOCK_C], dtype=tl.float32)
    partial_sq = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c0 in range(0, channels, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        mask = cols < channels
        value = tl.load(value_ptr + value_base + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + value_base + cols, mask=mask, other=0.0).to(tl.float32)
            value = value + residual
        partial += tl.where(mask, value, 0.0)
        partial_sq += tl.where(mask, value * value, 0.0)
    mean = tl.sum(partial) / channels
    variance = tl.maximum(tl.sum(partial_sq) / channels - mean * mean, 0.0)
    rstd = tl.rsqrt(variance + tl.cast(eps, tl.float32))
    for c0 in range(0, padded_channels, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        valid = cols < channels
        value = tl.load(value_ptr + value_base + cols, mask=valid, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + value_base + cols, mask=valid, other=0.0).to(tl.float32)
            value = value + residual
        out = (value - mean) * rstd
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + cols, mask=valid, other=0.0).to(tl.float32)
            out = out * weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=valid, other=0.0).to(tl.float32)
            out = out + bias
        out = tl.where(valid, out, 0.0)
        out = _round_to_dense(out, ROUND)
        _store_quantized(out_ptr, row, cols, out, scale_ptr, padded_channels)


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
):
    """Depthwise 5x5 + bias + ReLU + dynamic MXFP8 over a ``BW x TILE_H`` tile, 32 channels.

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
            amax = tl.max(tl.abs(activated), axis=1)
            exponent = tl.ceil(tl.log2(amax / 448.0))
            exponent = tl.maximum(tl.minimum(exponent, 127.0), -127.0)
            exponent = tl.where(amax == 0.0, -127.0, exponent)
            quantized = tl.maximum(tl.minimum(activated * tl.exp2(-exponent)[:, None], 448.0), -448.0)
            row = (batch * HEIGHT + out_y) * WIDTH + x
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


@triton.jit
def _fp8_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    bias_ptr,
    out_ptr,
    rows,
    N,
    K: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Static-scale activation x per-row-block weight FP8 GEMM, BF16 output."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        for k_tile in tl.static_range(BLOCK_K // 32):
            offs_k = k0 + k_tile * 32 + tl.arange(0, 32)
            kb = k0 // 32 + k_tile
            a_bits = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_m[:, None], other=0.0)
            a_scale = tl.exp2(tl.load(a_scale_ptr + kb).to(tl.float32))
            b_bits = tl.load(b_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_n[:, None], other=0.0)
            b_scale_exp = tl.load(b_scale_ptr + offs_n * K_BLOCKS + kb, mask=mask_n, other=-127)
            b_scale = tl.exp2(b_scale_exp.to(tl.float32))
            accumulator += tl.dot(a_bits, tl.trans(b_bits)) * (a_scale * b_scale[None, :])
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _empty_quantized(value: Tensor, padded: int) -> Tensor:
    return torch.empty((*value.shape[:-1], padded), device=value.device, dtype=torch.float8_e4m3fn)


def layernorm_quantize(
    value: Tensor,
    residual: Tensor | None,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
    scale_exponent: Tensor,
) -> QuantizedTensor:
    channels = value.shape[-1]
    padded = padded_channels(channels)
    out = _empty_quantized(value, padded)
    rows = value.numel() // channels
    block_c = min(1024, triton.next_power_of_2(channels))
    _layernorm_quantize_kernel[(rows,)](
        value,
        residual if residual is not None else value,
        weight.contiguous().reshape(-1) if weight is not None else value,
        bias.contiguous().reshape(-1) if bias is not None else value,
        scale_exponent,
        out,
        channels,
        padded,
        float(eps),
        BLOCK_C=block_c,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        ROUND=_ROUND_CODE[value.dtype],
        num_warps=4 if block_c >= 128 else 1,
    )
    return QuantizedTensor(out, scale_exponent, tuple(value.shape))


def gelu_quantize(value: Tensor, scale_exponent: Tensor, approximate: str) -> QuantizedTensor:
    channels = value.shape[-1]
    padded = padded_channels(channels)
    out = _empty_quantized(value, padded)
    rows = value.numel() // channels
    block = 1024
    _gelu_quantize_kernel[(triton.cdiv(rows * padded, block),)](
        value,
        scale_exponent,
        out,
        channels,
        padded,
        rows * padded,
        BLOCK=block,
        TANH=approximate == "tanh",
        ROUND=_ROUND_CODE[value.dtype],
        num_warps=4,
    )
    return QuantizedTensor(out, scale_exponent, tuple(value.shape))


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
        num_warps=num_warps,
    )
    return quantized, scale_bits.view(torch.float8_e8m0fnu)


def _gemm_config(rows: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    """Shape-deterministic launch config (RoMaV2 ``select_gemm_config``)."""
    block_k = 32 if rows <= 1024 else (64 if k % 64 == 0 else 32)
    if n >= 128 and rows >= 64:
        block_m = 64
    elif rows >= 128:
        block_m = 128
    elif rows >= 64:
        block_m = 64
    elif rows >= 32:
        block_m = 32
    else:
        block_m = 16
    block_n = 128 if n >= 128 else (64 if n >= 64 else 32)
    return block_m, block_n, block_k, 4, 4 if k >= 2048 else 3


def fp8_gemm_bf16(activation: QuantizedTensor, weight: QuantizedWeight, bias: Tensor | None) -> Tensor:
    """``[rows, K] x [N, K]^T`` with static activation scales; padded BF16 output."""
    matrix = activation.matrix().contiguous()
    rows = matrix.shape[0]
    n, k = weight.out_features, weight.in_features
    out = torch.empty(rows, n, device=matrix.device, dtype=torch.bfloat16)
    block_m, block_n, block_k, num_warps, num_stages = _gemm_config(rows, k, n)
    _fp8_gemm_kernel[(triton.cdiv(rows, block_m) * triton.cdiv(n, block_n),)](
        matrix,
        activation.scale_exponent,
        weight.values,
        weight.scale_exponent,
        bias if bias is not None else out,
        out,
        rows,
        n,
        K=k,
        K_BLOCKS=k // 32,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


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
