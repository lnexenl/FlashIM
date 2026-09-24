"""MXFP8 math: E4M3 values with one power-of-two (E8M0) scale per 32 elements.

Scale exponents are stored as unbiased ``int8`` (``2**e``); ``e = -127`` marks an
all-zero block. ``F.scaled_mm`` consumes E8M0 bytes (``e + 127``) in the
cuBLAS ``SWIZZLE_32_4_4`` layout produced by :func:`swizzle_scales`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

FP8_MAX = 448.0
BLOCK = 32


def padded_channels(channels: int) -> int:
    return ((channels + BLOCK - 1) // BLOCK) * BLOCK


def scale_exponent_from_amax(amax: Tensor) -> Tensor:
    """Smallest power of two ``2**e`` such that ``amax / 2**e <= 448``."""
    exponent = torch.ceil(torch.log2(amax / FP8_MAX))
    exponent = torch.where(amax == 0, torch.full_like(exponent, -127), exponent)
    return exponent.clamp(-127, 127).to(torch.int8)


def quantize_blocks(blocks: Tensor, scale_exponent: Tensor) -> Tensor:
    """Divide ``[..., groups, 32]`` blocks by their scales and saturate (FP32 result)."""
    scales = torch.exp2(scale_exponent.float())
    if scale_exponent.ndim == 1:
        scales = scales[None, :, None]
    elif scale_exponent.ndim == 2:
        scales = scales[..., None]
    else:
        raise ValueError(f"Unsupported scale rank: {scale_exponent.ndim}")
    return (blocks / scales).clamp(-FP8_MAX, FP8_MAX)


def calibrate_block32_exponent(value: Tensor) -> Tensor:
    """One static exponent per padded 32-channel block, max over all rows."""
    if value.ndim < 2:
        raise ValueError(f"Expected a channelized tensor, got {tuple(value.shape)}")
    padded = padded_channels(value.shape[-1])
    blocks = F.pad(value.float(), (0, padded - value.shape[-1])).reshape(-1, padded // BLOCK, BLOCK)
    return scale_exponent_from_amax(blocks.abs().amax(dim=(0, 2)))


def quantize_dynamic_2d(value: Tensor) -> tuple[Tensor, Tensor]:
    """Per-(row, 32-block) dynamic MXFP8 quantization of a ``[rows, K]`` matrix."""
    if value.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(value.shape)}")
    rows, cols = value.shape
    if cols % BLOCK != 0:
        raise ValueError(f"MXFP8 requires K divisible by 32, got K={cols}")
    blocks = value.float().contiguous().reshape(rows, cols // BLOCK, BLOCK)
    exponent = scale_exponent_from_amax(blocks.abs().amax(dim=-1))
    quantized = (blocks * torch.exp2(-exponent.float())[..., None]).clamp(-FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).reshape(rows, cols), exponent


def exponent_to_e8m0(exponent: Tensor) -> Tensor:
    return (exponent.to(torch.int16) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def swizzle_scales(exponent: Tensor) -> Tensor:
    """Pack ``[rows, K/32]`` int8 exponents into cuBLAS' 128x4 swizzled E8M0 layout."""
    if exponent.ndim != 2:
        raise ValueError(f"Expected [rows, groups] exponents, got {tuple(exponent.shape)}")
    scales = exponent_to_e8m0(exponent)
    rows, cols = scales.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    if rows != padded_rows or cols != padded_cols:
        padded = torch.zeros((padded_rows, padded_cols), device=scales.device, dtype=torch.uint8)
        padded[:rows, :cols] = scales.view(torch.uint8)
        scales = padded.view(torch.float8_e8m0fnu)
    return (
        scales.view(padded_rows // 128, 128, padded_cols // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
        .contiguous()
    )


def static_input_scales(exponent: Tensor, rows: int) -> Tensor:
    """Broadcast a static ``[K/32]`` exponent over ``rows`` and swizzle it."""
    if exponent.ndim != 1:
        raise ValueError("Expected a 1D static block-32 exponent")
    if rows <= 0:
        raise ValueError(f"Expected a positive row count, got {rows}")
    return swizzle_scales(exponent.to(torch.int8).expand(rows, -1))


def swizzled_scale_numel(rows: int, k: int) -> int:
    return ((rows + 127) // 128) * 128 * (((k // BLOCK + 3) // 4) * 4)
