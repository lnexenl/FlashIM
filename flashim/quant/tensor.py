"""Quantized tensor containers shared by quant methods, kernels and fused ops."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.quant.formats.int8 import quantize_symmetric
from flashim.quant.formats.mx import BLOCK, padded_channels, quantize_blocks, scale_exponent_from_amax


@dataclass(frozen=True)
class QuantizedTensor:
    """E4M3 activations with one static exponent per 32-channel block.

    ``values`` has shape ``[*leading, padded_channels]``; ``logical_shape`` is the
    unpadded dense shape. Scales are shared by every row (static calibration).
    """

    values: Tensor
    scale_exponent: Tensor
    logical_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.values.dtype != torch.float8_e4m3fn:
            raise ValueError(f"Expected float8_e4m3fn values, got {self.values.dtype}")
        if self.scale_exponent.dtype != torch.int8:
            raise ValueError("scale_exponent must be int8 unbiased exponents")
        channels = self.values.shape[-1]
        if channels % BLOCK or channels < self.logical_shape[-1]:
            raise ValueError(f"Bad physical channels {channels} for {self.logical_shape}")
        if tuple(self.scale_exponent.shape) != (channels // BLOCK,):
            raise ValueError(f"Expected scale shape {(channels // BLOCK,)}")

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def channels(self) -> int:
        return self.values.shape[-1]

    @property
    def rows(self) -> int:
        return math.prod(self.values.shape[:-1])

    def matrix(self) -> Tensor:
        return self.values.reshape(-1, self.channels)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        blocks = self.values.reshape(self.rows, self.channels // BLOCK, BLOCK).float()
        dense = (blocks * torch.exp2(self.scale_exponent.float())[..., None]).reshape(self.rows, -1)
        return dense[:, : self.logical_shape[-1]].reshape(self.logical_shape).to(dtype)


@dataclass(frozen=True)
class QuantizedWeight:
    """E4M3 ``[N, K]`` weight with one exponent per (output row, 32 inputs)."""

    values: Tensor
    scale_exponent: Tensor
    logical_shape: tuple[int, int]

    @property
    def out_features(self) -> int:
        return self.values.shape[0]

    @property
    def in_features(self) -> int:
        return self.values.shape[1]

    @classmethod
    def from_float(cls, weight: Tensor) -> QuantizedWeight:
        if weight.ndim != 2:
            raise ValueError(f"Unsupported weight shape: {tuple(weight.shape)}")
        if not torch.isfinite(weight).all():
            raise ValueError("Cannot quantize non-finite weights")
        out_padded, in_padded = padded_channels(weight.shape[0]), padded_channels(weight.shape[1])
        padded = F.pad(weight.float(), (0, in_padded - weight.shape[1], 0, out_padded - weight.shape[0]))
        blocks = padded.reshape(out_padded, in_padded // BLOCK, BLOCK)
        exponent = scale_exponent_from_amax(blocks.abs().amax(dim=-1))
        quantized = quantize_blocks(blocks, exponent).reshape(out_padded, in_padded)
        return cls(quantized.to(torch.float8_e4m3fn), exponent, tuple(weight.shape))

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        blocks = self.values.reshape(self.out_features, self.in_features // BLOCK, BLOCK).float()
        dense = (blocks * torch.exp2(self.scale_exponent.float())[..., None]).reshape(
            self.out_features, self.in_features
        )
        return dense[: self.logical_shape[0], : self.logical_shape[1]].to(dtype)


def quantize_static(value: Tensor, scale_exponent: Tensor) -> QuantizedTensor:
    """Reference static block-32 quantization (pads channels to a multiple of 32).

    ``value`` is made contiguous first: a non-contiguous input (e.g. the transposed view
    ``sample_descriptors`` produces) otherwise survives the ``.float()``/``.reshape()`` below as an
    equally non-contiguous, oddly-strided ``QuantizedTensor.values`` whenever the reshape happens to
    need no copy -- cuBLAS's block-scaled MXFP8 GEMM then rejects that layout with
    ``CUBLAS_STATUS_NOT_SUPPORTED`` instead of a shape error, only on the (non-Triton-eligible) reference
    path, so it was rare and looked data-dependent.
    """
    if value.ndim < 2:
        raise ValueError(f"Expected a channelized tensor, got {tuple(value.shape)}")
    padded = padded_channels(value.shape[-1])
    scale_exponent = scale_exponent.to(device=value.device, dtype=torch.int8)
    if tuple(scale_exponent.shape) != (padded // BLOCK,):
        raise ValueError(f"Expected scale shape {(padded // BLOCK,)}, got {tuple(scale_exponent.shape)}")
    dense = value.contiguous().float()
    if padded != value.shape[-1]:
        dense = F.pad(dense, (0, padded - value.shape[-1]))
    quantized = quantize_blocks(dense.reshape(-1, padded // BLOCK, BLOCK), scale_exponent)
    return QuantizedTensor(
        quantized.reshape(*value.shape[:-1], padded).to(torch.float8_e4m3fn),
        scale_exponent,
        tuple(value.shape),
    )


@dataclass(frozen=True)
class Int8Tensor:
    """INT8 activations with one static per-tensor FP32 scale; ``values`` is ``[*leading, K]``."""

    values: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        if self.values.dtype != torch.int8:
            raise ValueError(f"Expected int8 values, got {self.values.dtype}")
        if self.scale.dtype != torch.float32 or self.scale.numel() != 1:
            raise ValueError("scale must be a single FP32 value")

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    def matrix(self) -> Tensor:
        return self.values.reshape(-1, self.values.shape[-1])

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        return (self.values.float() * self.scale).to(dtype)


def quantize_int8_static(value: Tensor, scale: Tensor) -> Int8Tensor:
    """Reference static per-tensor INT8 quantization."""
    scale = scale.to(device=value.device, dtype=torch.float32).reshape(())
    return Int8Tensor(quantize_symmetric(value, scale), scale)
