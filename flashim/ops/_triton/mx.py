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
def _depthwise5x5_relu_mxfp8_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    scale_bits_ptr,
    rows,
    rows_per_image,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHANNELS: tl.constexpr,
    SCALE_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    channel = group * 32 + tl.arange(0, 32)
    batch = row // rows_per_image
    image_row = row - batch * rows_per_image
    y = image_row // WIDTH
    x = image_row - y * WIDTH
    valid_row = row < rows
    batch_base = batch * rows_per_image * CHANNELS

    accumulator = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    for ky in range(5):
        input_y = y + ky - 2
        valid_y = (input_y >= 0) & (input_y < HEIGHT)
        for kx in range(5):
            input_x = x + kx - 2
            valid_x = (input_x >= 0) & (input_x < WIDTH)
            input_offset = (
                batch_base[:, None]
                + (input_y[:, None] * WIDTH + input_x[:, None]) * CHANNELS
                + channel[None, :]
            )
            values = (
                tl.load(
                    input_ptr + input_offset,
                    mask=valid_row[:, None] & valid_y[:, None] & valid_x[:, None],
                    other=0.0,
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            weight = tl.load(weight_ptr + (ky * 5 + kx) * CHANNELS + channel).to(tl.float32)
            accumulator += values * weight[None, :]

    bias = tl.load(bias_ptr + channel).to(tl.float32)
    accumulator = tl.maximum(accumulator + bias[None, :], 0.0)
    amax = tl.max(tl.abs(accumulator), axis=1)
    exponent = tl.ceil(tl.log2(amax / 448.0))
    exponent = tl.maximum(tl.minimum(exponent, 127.0), -127.0)
    exponent = tl.where(amax == 0.0, -127.0, exponent)
    quantized = tl.maximum(tl.minimum(accumulator * tl.exp2(-exponent)[:, None], 448.0), -448.0)
    tl.store(output_ptr + row[:, None] * CHANNELS + channel[None, :], quantized, mask=valid_row[:, None])

    row_in_tile = row % 128
    scale_offset = (
        ((row // 128) * SCALE_TILES + group // 4) * 512
        + (row_in_tile % 32) * 16
        + (row_in_tile // 32) * 4
        + group % 4
    )
    scale_bits = tl.where(valid_row, exponent + 127.0, 0.0).to(tl.uint8)
    tl.store(scale_bits_ptr + scale_offset, scale_bits)


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
    value: Tensor, weight: Tensor, bias: Tensor, *, block_m: int = 32, num_warps: int = 4
) -> tuple[Tensor, Tensor]:
    batch, channels, height, width = value.shape
    groups = channels // 32
    scale_tiles = (groups + 3) // 4
    rows_per_image = height * width
    rows = batch * rows_per_image
    padded_rows = triton.cdiv(rows, 128) * 128
    quantized = torch.empty((rows, channels), device=value.device, dtype=torch.float8_e4m3fn)
    scale_bits = torch.zeros(padded_rows * scale_tiles * 4, device=value.device, dtype=torch.uint8)
    _depthwise5x5_relu_mxfp8_kernel[(triton.cdiv(padded_rows, block_m), groups)](
        value,
        weight,
        bias,
        quantized,
        scale_bits,
        rows,
        rows_per_image,
        HEIGHT=height,
        WIDTH=width,
        BLOCK_M=block_m,
        CHANNELS=channels,
        SCALE_TILES=scale_tiles,
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
