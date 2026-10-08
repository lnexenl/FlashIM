# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/full_fp8.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_FP8_MAX = 448.0
ActivationKind = Literal["none", "gelu", "silu"]


def _allow_in_graph(function):
    compiler = getattr(torch, "compiler", None)
    allow_in_graph = getattr(compiler, "allow_in_graph", None)
    return function if allow_in_graph is None else allow_in_graph(function)


@_allow_in_graph
def _mxfp8_scaled_mm(
    matrix: Tensor, weight: Tensor, input_scale: Tensor, weight_scale: Tensor
) -> Tensor:
    # Dynamo can proxy tensors, but not the pybind ScalingType/SwizzleType
    # objects. Keep those constants inside the AOT-traced call boundary.
    return F.scaled_mm(
        matrix,
        weight,
        input_scale,
        F.ScalingType.BlockWise1x32,
        weight_scale,
        F.ScalingType.BlockWise1x32,
        F.SwizzleType.SWIZZLE_32_4_4,
        F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )


class QuantizedLayout(str, Enum):
    TOKEN_MATRIX = "token_matrix"
    CHANNELS_LAST = "channels_last"


@dataclass(frozen=True)
class QuantizedTensor:
    """FP8 activations plus one static E8M0 scale per block of 32 channels."""

    values: Tensor
    scale_exponent: Tensor
    logical_shape: tuple[int, ...]
    layout: QuantizedLayout = QuantizedLayout.TOKEN_MATRIX

    def __post_init__(self) -> None:
        if self.values.dtype != torch.float8_e4m3fn:
            raise ValueError(f"Expected float8_e4m3fn values, got {self.values.dtype}")
        if self.scale_exponent.dtype != torch.int8:
            raise ValueError("scale_exponent must use int8 unbiased exponents")
        if self.values.device != self.scale_exponent.device:
            raise ValueError("values and scale_exponent must be on the same device")
        if len(self.logical_shape) < 2:
            raise ValueError(f"Expected a channelized shape, got {self.logical_shape}")
        channels = self.values.shape[-1]
        logical_channels = self.logical_shape[-1]
        if channels % 32 != 0:
            raise ValueError(
                f"Quantized channels must be divisible by 32, got {channels}"
            )
        if channels < logical_channels:
            raise ValueError(
                f"Physical channels {channels} cannot be less than logical channels {logical_channels}"
            )
        expected_scale_shape = (channels // 32,)
        if tuple(self.scale_exponent.shape) != expected_scale_shape:
            raise ValueError(
                f"Expected scale shape {expected_scale_shape}, got {tuple(self.scale_exponent.shape)}"
            )
        value_rows = math.prod(self.values.shape[:-1])
        logical_rows = math.prod(self.logical_shape[:-1])
        if value_rows != logical_rows:
            raise ValueError(
                f"Expected {logical_rows} logical rows, got {value_rows} physical rows"
            )
        if (
            self.layout == QuantizedLayout.CHANNELS_LAST
            and len(self.logical_shape) != 4
        ):
            raise ValueError("channels-last activations must have a 4D logical shape")

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def channels(self) -> int:
        return self.values.shape[-1]

    def matrix(self) -> Tensor:
        return self.values.reshape(-1, self.channels)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        rows = math.prod(self.values.shape[:-1])
        blocks = self.values.reshape(rows, self.channels // 32, 32).float()
        scales = torch.exp2(self.scale_exponent.float())
        dequantized = (blocks * scales[..., None]).reshape(rows, self.channels)
        dequantized = dequantized[..., : self.logical_shape[-1]].reshape(
            self.logical_shape
        )
        return dequantized.to(dtype)

    def to(self, device: torch.device) -> QuantizedTensor:
        return QuantizedTensor(
            self.values.to(device),
            self.scale_exponent.to(device),
            self.logical_shape,
            self.layout,
        )


@dataclass(frozen=True)
class QuantizedWeight:
    """FP8 matrix weight with one E8M0 scale per output row and 32 inputs."""

    values: Tensor
    scale_exponent: Tensor
    logical_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.values.dtype != torch.float8_e4m3fn or self.values.ndim != 2:
            raise ValueError("Expected a 2D float8_e4m3fn weight")
        if self.scale_exponent.dtype != torch.int8:
            raise ValueError("scale_exponent must use int8 unbiased exponents")
        if self.values.device != self.scale_exponent.device:
            raise ValueError("values and scale_exponent must be on the same device")
        out_features, in_features = self.values.shape
        logical_out, logical_in = self.logical_shape
        if in_features % 32 != 0:
            raise ValueError(f"Physical K must be divisible by 32, got {in_features}")
        if out_features % 32 != 0:
            raise ValueError(f"Physical N must be divisible by 32, got {out_features}")
        if logical_in > in_features or logical_out > out_features:
            raise ValueError(
                f"Logical shape {self.logical_shape} exceeds physical shape {tuple(self.values.shape)}"
            )
        expected_scale_shape = (out_features, in_features // 32)
        if tuple(self.scale_exponent.shape) != expected_scale_shape:
            raise ValueError(
                f"Expected scale shape {expected_scale_shape}, got {tuple(self.scale_exponent.shape)}"
            )

    @property
    def device(self) -> torch.device:
        return self.values.device

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
        logical_shape = tuple(weight.shape)
        physical_out = ((weight.shape[0] + 31) // 32) * 32
        physical_in = ((weight.shape[1] + 31) // 32) * 32
        padded = F.pad(
            weight.float(),
            (0, physical_in - weight.shape[1], 0, physical_out - weight.shape[0]),
        )
        blocks = padded.reshape(physical_out, physical_in // 32, 32)
        exponent = _scale_exponent_from_amax(blocks.abs().amax(dim=-1))
        quantized = _quantize_blocks(blocks, exponent)
        return cls(
            quantized.reshape(physical_out, physical_in).to(torch.float8_e4m3fn),
            exponent,
            logical_shape,
        )

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        blocks = self.values.reshape(
            self.out_features, self.in_features // 32, 32
        ).float()
        scales = torch.exp2(self.scale_exponent.float())
        dequantized = (blocks * scales[..., None]).reshape(
            self.out_features, self.in_features
        )
        return dequantized[: self.logical_shape[0], : self.logical_shape[1]].to(dtype)

    def to(self, device: torch.device) -> QuantizedWeight:
        return QuantizedWeight(
            self.values.to(device),
            self.scale_exponent.to(device),
            self.logical_shape,
        )


def _scale_exponent_from_amax(amax: Tensor) -> Tensor:
    exponent = torch.ceil(torch.log2(amax / _FP8_MAX))
    exponent = torch.where(amax == 0, torch.full_like(exponent, -127), exponent)
    return exponent.clamp(-127, 127).to(torch.int8)


def calibrate_block32_scale(value: Tensor) -> Tensor:
    """Calibrate one static E8M0 exponent per padded block of 32 channels."""
    if value.ndim < 2:
        raise ValueError(f"Expected a channelized tensor, got {tuple(value.shape)}")
    padded_channels = ((value.shape[-1] + 31) // 32) * 32
    padded = F.pad(value.float(), (0, padded_channels - value.shape[-1]))
    blocks = padded.reshape(-1, padded_channels // 32, 32)
    return _scale_exponent_from_amax(blocks.abs().amax(dim=(0, 2)))


def _quantize_blocks(blocks: Tensor, scale_exponent: Tensor) -> Tensor:
    scales = torch.exp2(scale_exponent.float())
    if scale_exponent.ndim == 1:
        scales = scales[None, :, None]
    elif scale_exponent.ndim == 2:
        scales = scales[..., None]
    else:
        raise ValueError(f"Unsupported scale rank: {scale_exponent.ndim}")
    return (blocks / scales).clamp(-_FP8_MAX, _FP8_MAX)


def quantize_block32(
    value: Tensor,
    scale_exponent: Tensor,
    layout: QuantizedLayout = QuantizedLayout.TOKEN_MATRIX,
) -> QuantizedTensor:
    """Quantize a tensor with a caller-provided static padded block-32 scale."""
    if value.ndim < 2:
        raise ValueError(f"Expected a channelized tensor, got {tuple(value.shape)}")
    logical_shape = tuple(value.shape)
    padded_channels = ((value.shape[-1] + 31) // 32) * 32
    expected_scales = padded_channels // 32
    scale_exponent = scale_exponent.to(device=value.device, dtype=torch.int8)
    if tuple(scale_exponent.shape) != (expected_scales,):
        raise ValueError(
            f"Expected scale shape {(expected_scales,)}, got {tuple(scale_exponent.shape)}"
        )
    padding = padded_channels - value.shape[-1]
    padded = value.float() if padding == 0 else F.pad(value.float(), (0, padding))
    blocks = padded.reshape(-1, padded_channels // 32, 32)
    quantized = _quantize_blocks(blocks, scale_exponent)
    return QuantizedTensor(
        quantized.reshape(*value.shape[:-1], padded_channels).to(torch.float8_e4m3fn),
        scale_exponent,
        logical_shape,
        layout,
    )


def _swizzle_mxfp8_scale(scale: Tensor) -> Tensor:
    """Pack logical [rows, K/32] E8M0 scales for SM120 block-scaled GEMMs."""
    if scale.ndim != 2 or scale.dtype != torch.float8_e8m0fnu:
        raise ValueError("Expected a 2D float8_e8m0fnu MXFP8 scale tensor")
    rows, groups = scale.shape
    padded_rows = ((rows + 127) // 128) * 128
    macro_columns = (groups + 3) // 4
    row = torch.arange(rows, device=scale.device)[:, None]
    column = torch.arange(groups, device=scale.device)[None, :]
    offset = (
        ((row // 128) * macro_columns + column // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + column % 4
    )
    packed = torch.zeros(
        padded_rows * macro_columns * 4, device=scale.device, dtype=torch.uint8
    )
    packed[offset.reshape(-1)] = scale.reshape(-1).view(torch.uint8)
    return packed.view(torch.float8_e8m0fnu)


def _mxfp8_input_scale(scale_exponent: Tensor, rows: int) -> Tensor:
    """Expand static block-32 scales over rows and pack cuBLAS' 128x4 layout."""
    if rows <= 0:
        raise ValueError(f"Expected positive row count, got {rows}")
    if scale_exponent.ndim != 1 or scale_exponent.dtype != torch.int8:
        raise ValueError("Expected a 1D int8 static block-32 scale exponent")
    scale = torch.exp2(scale_exponent.float()).to(torch.float8_e8m0fnu)
    return _swizzle_mxfp8_scale(scale.expand(rows, -1))


def fake_quantize_block32(value: Tensor, scale_exponent: Tensor) -> Tensor:
    """Apply QAT-style quantize/dequantize without changing tensor dtype."""
    return quantize_block32(value, scale_exponent).dequantize(dtype=value.dtype)


class FakeQuantizedLinear(nn.Module):
    """QAT wrapper that fake-quantizes inputs, weights, and linear outputs."""

    def __init__(
        self,
        linear: nn.Linear,
        input_scale_exponent: Tensor,
        output_scale_exponent: Tensor,
        activation: ActivationKind = "none",
    ) -> None:
        super().__init__()
        if activation not in ("none", "gelu", "silu"):
            raise ValueError(f"Unsupported activation: {activation}")
        self.linear = linear
        self.activation = activation
        self.register_buffer(
            "input_scale_exponent",
            input_scale_exponent.detach().to(dtype=torch.int8).clone(),
        )
        self.register_buffer(
            "output_scale_exponent",
            output_scale_exponent.detach().to(dtype=torch.int8).clone(),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = fake_quantize_block32(value, self.input_scale_exponent)
        weight = QuantizedWeight.from_float(self.linear.weight).dequantize(
            dtype=self.linear.weight.dtype
        )
        output = F.linear(value, weight, self.linear.bias)
        if self.activation == "gelu":
            output = F.gelu(output)
        elif self.activation == "silu":
            output = F.silu(output)
        return fake_quantize_block32(output, self.output_scale_exponent)


if triton is not None:

    @triton.jit
    def _fp8_gemm_fp8_out_kernel(
        a_ptr,
        a_scale_ptr,
        b_ptr,
        b_scale_ptr,
        bias_ptr,
        residual_ptr,
        residual_scale_ptr,
        out_ptr,
        out_scale_ptr,
        rows,
        N,
        K: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        ACTIVATION: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
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

                a_bits = tl.load(
                    a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=mask_m[:, None],
                    other=0.0,
                )
                a_scale_exp = tl.load(a_scale_ptr + kb)
                a_scale = tl.exp2(a_scale_exp.to(tl.float32))

                b_bits = tl.load(
                    b_ptr + offs_n[:, None] * K + offs_k[None, :],
                    mask=mask_n[:, None],
                    other=0.0,
                )
                b_scale_exp = tl.load(
                    b_scale_ptr + offs_n * K_BLOCKS + kb, mask=mask_n, other=-127
                )
                b_scale = tl.exp2(b_scale_exp.to(tl.float32))
                accumulator += tl.dot(a_bits, tl.trans(b_bits)) * (
                    a_scale * b_scale[None, :]
                )

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            accumulator += bias[None, :]

        if HAS_RESIDUAL:
            residual_bits = tl.load(
                residual_ptr + offs_m[:, None] * N + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            residual_scale_exp = tl.load(
                residual_scale_ptr + offs_n // 32, mask=mask_n, other=-127
            )
            residual_scale = tl.exp2(residual_scale_exp.to(tl.float32))
            accumulator += residual_bits.to(tl.float32) * residual_scale[None, :]

        if ACTIVATION == 1:
            accumulator = (
                0.5
                * accumulator
                * (1.0 + tl.math.erf(accumulator * 0.7071067811865476))
            )
        elif ACTIVATION == 2:
            accumulator = accumulator * tl.sigmoid(accumulator)

        out_scale_exp = tl.load(out_scale_ptr + offs_n // 32, mask=mask_n, other=-127)
        inverse_scale = tl.exp2(-out_scale_exp.to(tl.float32))
        quantized = tl.maximum(
            tl.minimum(accumulator * inverse_scale[None, :], 448.0), -448.0
        ).to(tl.float8e4nv)
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            quantized,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _fp8_gemm_fp32_out_kernel(
        a_ptr,
        a_scale_ptr,
        b_ptr,
        b_scale_ptr,
        out_ptr,
        rows,
        N,
        K: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
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
                a_bits = tl.load(
                    a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=mask_m[:, None],
                    other=0.0,
                )
                a_scale_exp = tl.load(a_scale_ptr + kb)
                b_bits = tl.load(
                    b_ptr + offs_n[:, None] * K + offs_k[None, :],
                    mask=mask_n[:, None],
                    other=0.0,
                )
                b_scale_exp = tl.load(
                    b_scale_ptr + offs_n * K_BLOCKS + kb,
                    mask=mask_n,
                    other=-127,
                )
                scale = tl.exp2(
                    a_scale_exp.to(tl.float32) + b_scale_exp.to(tl.float32)[None, :]
                )
                accumulator += tl.dot(a_bits, tl.trans(b_bits)) * scale

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            accumulator,
            mask=mask_m[:, None] & mask_n[None, :],
        )

else:
    _fp8_gemm_fp8_out_kernel = None
    _fp8_gemm_fp32_out_kernel = None


def _activation_code(activation: ActivationKind) -> int:
    return {"none": 0, "gelu": 1, "silu": 2}[activation]


@dataclass(frozen=True)
class GemmConfig:
    """Triton launch configuration for the FP8 GEMM kernels.

    The defaults reproduce the historical fixed launch parameters.
    """

    block_m: int = 128
    block_n: int = 64
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 3


def select_gemm_config(rows: int, k: int, n: int) -> GemmConfig:
    """Deterministically select an SM120 launch config for an (M, K, N) GEMM.

    The selection is a pure function of the matrix shape; it never queries the
    device. ``block_k`` always divides ``k`` (the kernel K loop has no tail
    mask), and every block dimension stays within ``tl.dot`` limits.

    Covers the production shape families: small-M token batches, large-N QKV
    projections, MLP up/down projections, and narrow pointwise (1x1 conv)
    GEMMs.
    """
    if rows <= 0 or k <= 0 or n <= 0:
        raise ValueError(f"Expected positive GEMM dims, got ({rows}, {k}, {n})")
    if k % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {k}")

    # The kernel K loop has no tail mask: block_k must divide K exactly.
    # At the 800-token DINO/matcher workloads, a 32-wide K tile reduces
    # register pressure enough to beat the wider reduction. Larger refiner
    # matrices retain 64-wide K tiles.
    block_k = 32 if rows <= 1_024 else (64 if k % 64 == 0 else 32)

    # The 128x64 FP32 accumulator needs 255 registers/thread on SM120. For
    # wide-output transformer/refiner projections, 64x128 doubles the grid
    # residency while keeping the output tile fully utilized. Narrow
    # pointwise projections retain the wider M tile to avoid wasting N lanes.
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

    if n >= 128:
        block_n = 128
    elif n >= 64:
        block_n = 64
    else:
        block_n = 32

    # Eight warps amplify register pressure for the 128-wide N tile on this
    # kernel; the measured production shapes are faster with four.
    num_warps = 4
    # Deep-K reductions (MLP down-projections) benefit from an extra stage.
    num_stages = 4 if k >= 2048 else 3
    return GemmConfig(block_m, block_n, block_k, num_warps, num_stages)


def fp8_gemm(
    activation: QuantizedTensor,
    weight: QuantizedWeight,
    output_scale_exponent: Tensor,
    *,
    bias: Tensor | None = None,
    residual: QuantizedTensor | None = None,
    activation_kind: ActivationKind = "none",
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> QuantizedTensor:
    """Run an FP8-input GEMM whose epilogue directly emits FP8.

    Omitted launch arguments are dispatched by :func:`select_gemm_config`;
    explicitly passed values are used as-is.
    """
    if triton is None or _fp8_gemm_fp8_out_kernel is None:
        raise RuntimeError("The full-FP8 GEMM requires Triton")
    if activation.device.type != "cuda":
        raise RuntimeError("The full-FP8 GEMM requires CUDA activations")
    if weight.device != activation.device:
        raise ValueError("activation and weight must be on the same device")
    if activation.channels != weight.in_features:
        raise ValueError(
            f"Input width {activation.channels} does not match weight K {weight.in_features}"
        )
    output_scale_exponent = output_scale_exponent.to(
        device=activation.device, dtype=torch.int8
    )
    expected_output_scales = (weight.out_features // 32,)
    if tuple(output_scale_exponent.shape) != expected_output_scales:
        raise ValueError(
            f"Expected output scale shape {expected_output_scales}, got {tuple(output_scale_exponent.shape)}"
        )
    if bias is not None:
        if bias.device != activation.device or bias.dtype != torch.float32:
            raise ValueError(
                "bias must be a preconverted FP32 tensor on the activation device"
            )
        if tuple(bias.shape) != (weight.out_features,):
            raise ValueError(
                f"Expected padded bias shape {(weight.out_features,)}, got {tuple(bias.shape)}"
            )
    if residual is not None:
        if residual.device != activation.device:
            raise ValueError("residual must be on the activation device")
        if residual.channels != weight.out_features:
            raise ValueError(
                f"Residual width {residual.channels} does not match output width {weight.out_features}"
            )
        if residual.matrix().shape[0] != activation.matrix().shape[0]:
            raise ValueError("Residual and activation row counts do not match")

    matrix = activation.matrix().contiguous()
    rows = matrix.shape[0]
    output = torch.empty(
        rows,
        weight.out_features,
        device=activation.device,
        dtype=torch.float8_e4m3fn,
    )
    if (
        block_m is None
        or block_n is None
        or block_k is None
        or num_warps is None
        or num_stages is None
    ):
        selected = select_gemm_config(rows, k=weight.in_features, n=weight.out_features)
        block_m = selected.block_m if block_m is None else block_m
        block_n = selected.block_n if block_n is None else block_n
        block_k = selected.block_k if block_k is None else block_k
        num_warps = selected.num_warps if num_warps is None else num_warps
        num_stages = selected.num_stages if num_stages is None else num_stages
    if block_k % 32 != 0:
        raise ValueError(f"block_k must be divisible by 32, got {block_k}")
    if weight.in_features % block_k != 0:
        raise ValueError(
            f"block_k {block_k} must divide input width {weight.in_features}"
        )
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(weight.out_features, block_n),)
    _fp8_gemm_fp8_out_kernel[grid](
        matrix,
        activation.scale_exponent,
        weight.values,
        weight.scale_exponent,
        bias if bias is not None else output,
        residual.matrix().contiguous() if residual is not None else output,
        residual.scale_exponent if residual is not None else output_scale_exponent,
        output,
        output_scale_exponent,
        rows,
        weight.out_features,
        K=weight.in_features,
        K_BLOCKS=weight.in_features // 32,
        HAS_BIAS=bias is not None,
        HAS_RESIDUAL=residual is not None,
        ACTIVATION=_activation_code(activation_kind),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    logical_shape = (*activation.logical_shape[:-1], weight.logical_shape[0])
    return QuantizedTensor(
        output.reshape(*activation.values.shape[:-1], weight.out_features),
        output_scale_exponent,
        logical_shape,
        activation.layout,
    )


def fp8_gemm_fp32(
    activation: QuantizedTensor,
    weight: QuantizedWeight,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Tensor:
    """Run an FP8-input GEMM with FP32 accumulation and FP32 output.

    Omitted launch arguments are dispatched by :func:`select_gemm_config`;
    explicitly passed values are used as-is.
    """
    if triton is None or _fp8_gemm_fp32_out_kernel is None:
        raise RuntimeError("The full-FP8 GEMM requires Triton")
    if activation.device.type != "cuda":
        raise RuntimeError("The full-FP8 GEMM requires CUDA activations")
    if weight.device != activation.device:
        raise ValueError("activation and weight must be on the same device")
    if activation.channels != weight.in_features:
        raise ValueError(
            f"Input width {activation.channels} does not match weight K {weight.in_features}"
        )
    matrix = activation.matrix().contiguous()
    rows = matrix.shape[0]
    output = torch.empty(
        rows,
        weight.out_features,
        device=activation.device,
        dtype=torch.float32,
    )
    if (
        block_m is None
        or block_n is None
        or block_k is None
        or num_warps is None
        or num_stages is None
    ):
        selected = select_gemm_config(rows, k=weight.in_features, n=weight.out_features)
        block_m = selected.block_m if block_m is None else block_m
        block_n = selected.block_n if block_n is None else block_n
        block_k = selected.block_k if block_k is None else block_k
        num_warps = selected.num_warps if num_warps is None else num_warps
        num_stages = selected.num_stages if num_stages is None else num_stages
    if block_k % 32 != 0:
        raise ValueError(f"block_k must be divisible by 32, got {block_k}")
    if weight.in_features % block_k != 0:
        raise ValueError(
            f"block_k {block_k} must divide input width {weight.in_features}"
        )
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(weight.out_features, block_n),)
    _fp8_gemm_fp32_out_kernel[grid](
        matrix,
        activation.scale_exponent,
        weight.values,
        weight.scale_exponent,
        output,
        rows,
        weight.out_features,
        K=weight.in_features,
        K_BLOCKS=weight.in_features // 32,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    logical_shape = (*activation.logical_shape[:-1], weight.logical_shape[0])
    return output[:, : weight.logical_shape[0]].reshape(logical_shape)


class StaticFP8Linear(nn.Module):
    """Inference linear consuming and producing persistent block-32 FP8 tensors."""

    def __init__(
        self,
        weight: QuantizedWeight,
        bias: Tensor | None,
        output_scale_exponent: Tensor,
        *,
        input_scale_exponent: Tensor | None = None,
        activation: ActivationKind = "none",
    ) -> None:
        super().__init__()
        if activation not in ("none", "gelu", "silu"):
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activation
        self.logical_shape = weight.logical_shape
        self.register_buffer("weight_values", weight.values.view(torch.uint8))
        self.register_buffer("weight_scale_exponent", weight.scale_exponent)
        self.register_buffer(
            "weight_mxfp8_scale",
            _swizzle_mxfp8_scale(
                torch.exp2(weight.scale_exponent.float()).to(torch.float8_e8m0fnu)
            ),
        )
        self.register_buffer(
            "bias",
            None
            if bias is None
            else bias.detach().to(device=weight.device, dtype=torch.float32).clone(),
        )
        self.register_buffer(
            "mxfp8_bias",
            None
            if bias is None
            else bias.detach().to(device=weight.device, dtype=torch.bfloat16).clone(),
        )
        self.register_buffer(
            "output_scale_exponent",
            output_scale_exponent.detach()
            .to(device=weight.device, dtype=torch.int8)
            .clone(),
        )
        self.register_buffer(
            "input_scale_exponent",
            None
            if input_scale_exponent is None
            else input_scale_exponent.detach()
            .to(device=weight.device, dtype=torch.int8)
            .clone(),
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        output_scale_exponent: Tensor,
        *,
        input_scale_exponent: Tensor | None = None,
        activation: ActivationKind = "none",
    ) -> StaticFP8Linear:
        weight = QuantizedWeight.from_float(linear.weight.detach())
        bias = None
        if linear.bias is not None:
            bias = F.pad(
                linear.bias.detach(),
                (0, weight.out_features - linear.out_features),
            )
        return cls(
            weight,
            bias,
            output_scale_exponent,
            input_scale_exponent=input_scale_exponent,
            activation=activation,
        )

    @property
    def in_features(self) -> int:
        return self.logical_shape[1]

    @property
    def out_features(self) -> int:
        return self.logical_shape[0]

    def _weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            self.weight_values.view(torch.float8_e4m3fn),
            self.weight_scale_exponent,
            self.logical_shape,
        )

    @property
    def supports_native_mxfp8(self) -> bool:
        """Whether cuBLAS' SM120 MXFP8 kernel supports this padded weight."""
        return (
            self.weight_values.shape[0] % 128 == 0
            and self.weight_values.shape[1] % 128 == 0
        )

    def native_mxfp8(self, value: QuantizedTensor, input_scale: Tensor) -> Tensor:
        """Run an SM120 block-scaled FP8 GEMM and return padded BF16 output."""
        if not self.supports_native_mxfp8:
            raise ValueError(
                "Native MXFP8 requires padded K/N dimensions divisible by 128"
            )
        matrix = value.matrix().contiguous()
        rows = matrix.shape[0]
        expected_scale_elements = (
            ((rows + 127) // 128)
            * 128
            * (((self.weight_values.shape[1] // 32 + 3) // 4) * 4)
        )
        if (
            input_scale.dtype != torch.float8_e8m0fnu
            or input_scale.numel() != expected_scale_elements
        ):
            raise ValueError(
                "Native MXFP8 input scale has an incompatible dtype or packed size: "
                f"got {input_scale.dtype} with {input_scale.numel()} elements, "
                f"expected {expected_scale_elements} float8_e8m0fnu elements"
            )
        return _mxfp8_scaled_mm(
            matrix,
            self.weight_values.view(torch.float8_e4m3fn).transpose(0, 1),
            input_scale,
            self.weight_mxfp8_scale,
        )

    def forward(self, value: Tensor | QuantizedTensor) -> QuantizedTensor:
        if isinstance(value, Tensor):
            if self.input_scale_exponent is None:
                raise ValueError("A static input scale is required for dense inputs")
            value = quantize_block32(value, self.input_scale_exponent)
        return fp8_gemm(
            value,
            self._weight(),
            self.output_scale_exponent,
            bias=self.bias,
            activation_kind=self.activation,
        )
