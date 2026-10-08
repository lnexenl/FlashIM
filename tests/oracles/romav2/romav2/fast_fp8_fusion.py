# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/fast_fp8_fusion.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Fused residual-add + normalization/activation helpers for FastFP8 blocks.

Explicit, forward-only primitives for the static transformer blocks: a fused
``(value + residual) -> LayerNorm`` and a fused ``(value + residual) -> GELU``.
The op order matches the FP8 GEMM epilogue in :mod:`romav2.full_fp8` — the
residual add happens *before* the normalization/activation.

Numerical contract
------------------
1. The residual sum, LayerNorm statistics (mean/variance), and the GELU are
   computed in FP32 regardless of the input dtype; outputs are cast back to
   the input dtype at the end.
2. Outputs preserve the input shape, dtype, and device.
3. The Triton path is used for CUDA tensors in FP16/BF16/FP32 whose last
   dimension is contiguous; every other layout, rank, device, or dtype falls
   back to a direct torch implementation with identical math.

These are plain functions with explicit arguments — no module discovery, no
dispatch interception, no global state.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from romav2.full_fp8 import QuantizedTensor, quantize_block32

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

__all__ = [
    "fused_residual_gelu",
    "fused_residual_layernorm",
    "gelu_quantize_block32",
    "layernorm_quantize_block32",
    "residual_layernorm_quantize_block32",
]


_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
        value_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C,
        eps,
        BLOCK_C: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        base = row * C

        partial = tl.zeros([BLOCK_C], dtype=tl.float32)
        partial_sq = tl.zeros([BLOCK_C], dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            cols = c0 + tl.arange(0, BLOCK_C)
            mask = cols < C
            value = tl.load(value_ptr + base + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            residual = tl.load(residual_ptr + base + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            summed = value + residual
            partial += tl.where(mask, summed, 0.0)
            partial_sq += tl.where(mask, summed * summed, 0.0)

        mean = tl.sum(partial) / C
        variance = tl.maximum(tl.sum(partial_sq) / C - mean * mean, 0.0)
        rstd = 1.0 / tl.sqrt(variance + eps)

        for c0 in range(0, C, BLOCK_C):
            cols = c0 + tl.arange(0, BLOCK_C)
            mask = cols < C
            value = tl.load(value_ptr + base + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            residual = tl.load(residual_ptr + base + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            out = (value + residual - mean) * rstd
            if HAS_WEIGHT:
                weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                out = out * weight
            if HAS_BIAS:
                bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                out = out + bias
            tl.store(
                out_ptr + base + cols,
                out.to(out_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.jit
    def _residual_gelu_kernel(
        value_ptr,
        residual_ptr,
        out_ptr,
        numel,
        BLOCK: tl.constexpr,
        TANH: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        value = tl.load(value_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        summed = value + residual
        if TANH:
            # sqrt(2/pi) * (x + 0.044715 x^3); tanh(x) = 1 - 2/(exp(2x) + 1)
            # is stable at both infinities.
            inner = 0.7978845608028654 * (summed + 0.044715 * summed * summed * summed)
            tanh = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
            out = 0.5 * summed * (1.0 + tanh)
        else:
            # erf(x / sqrt(2)) form matching torch's exact GELU.
            out = 0.5 * summed * (1.0 + tl.math.erf(summed * 0.7071067811865476))
        tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _round_to_dense(value, ROUND: tl.constexpr):
        # Match quantize_block32's input: the dense op rounded to its dtype,
        # then upcast back to FP32 for the scale divide.
        if ROUND == 1:
            return value.to(tl.bfloat16).to(tl.float32)
        if ROUND == 2:
            return value.to(tl.float16).to(tl.float32)
        return value.to(tl.float32)

    @triton.jit
    def _store_quantized(
        out_ptr, row, cols, activated, scale_ptr, padded_channels, valid=True
    ):
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
        value = tl.load(
            value_ptr + row * channels + cols,
            mask=valid & (cols < channels),
            other=0.0,
        ).to(tl.float32)
        if TANH:
            inner = 0.7978845608028654 * (value + 0.044715 * value * value * value)
            tanh = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
            activated = 0.5 * value * (1.0 + tanh)
        else:
            activated = 0.5 * value * (1.0 + tl.math.erf(value * 0.7071067811865476))
        activated = _round_to_dense(activated, ROUND)
        _store_quantized(
            out_ptr, row, cols, activated, scale_ptr, padded_channels, valid
        )

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
            value = tl.load(value_ptr + value_base + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            if HAS_RESIDUAL:
                residual = tl.load(
                    residual_ptr + value_base + cols, mask=mask, other=0.0
                ).to(tl.float32)
                value = value + residual
            partial += tl.where(mask, value, 0.0)
            partial_sq += tl.where(mask, value * value, 0.0)
        mean = tl.sum(partial) / channels
        variance = tl.maximum(tl.sum(partial_sq) / channels - mean * mean, 0.0)
        rstd = tl.rsqrt(variance + tl.cast(eps, tl.float32))
        for c0 in range(0, padded_channels, BLOCK_C):
            cols = c0 + tl.arange(0, BLOCK_C)
            valid = cols < channels
            value = tl.load(value_ptr + value_base + cols, mask=valid, other=0.0).to(
                tl.float32
            )
            if HAS_RESIDUAL:
                residual = tl.load(
                    residual_ptr + value_base + cols, mask=valid, other=0.0
                ).to(tl.float32)
                value = value + residual
            out = (value - mean) * rstd
            if HAS_WEIGHT:
                weight = tl.load(weight_ptr + cols, mask=valid, other=0.0).to(
                    tl.float32
                )
                out = out * weight
            if HAS_BIAS:
                bias = tl.load(bias_ptr + cols, mask=valid, other=0.0).to(tl.float32)
                out = out + bias
            out = tl.where(valid, out, 0.0)
            out = _round_to_dense(out, ROUND)
            _store_quantized(out_ptr, row, cols, out, scale_ptr, padded_channels)


def _check_pair(op: str, value: Tensor, residual: Tensor) -> None:
    if not isinstance(value, Tensor) or not isinstance(residual, Tensor):
        raise TypeError(f"{op}: value and residual must be torch.Tensor")
    if value.shape != residual.shape:
        raise ValueError(
            f"{op}: value and residual must have the same shape, got "
            f"{tuple(value.shape)} and {tuple(residual.shape)}"
        )
    if value.dtype != residual.dtype:
        raise ValueError(
            f"{op}: value and residual must have the same dtype, got "
            f"{value.dtype} and {residual.dtype}"
        )
    if not value.is_floating_point():
        raise ValueError(f"{op}: inputs must use a floating dtype, got {value.dtype}")
    if value.device != residual.device:
        raise ValueError(
            f"{op}: value and residual must be on the same device, got "
            f"{value.device} and {residual.device}"
        )


def _check_affine(
    op: str, name: str, tensor: Tensor | None, normalized_shape: tuple[int, ...]
) -> None:
    if tensor is None:
        return
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{op}: {name} must be a torch.Tensor or None")
    if tuple(tensor.shape) != normalized_shape:
        raise ValueError(
            f"{op}: {name} must have shape {normalized_shape}, got "
            f"{tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise ValueError(f"{op}: {name} must use a floating dtype, got {tensor.dtype}")


def _triton_ready(value: Tensor, residual: Tensor) -> bool:
    """Whether the Triton path applies (CUDA, supported dtype, last-dim contig)."""
    return (
        triton is not None
        and value.is_cuda
        and value.dtype in _TRITON_DTYPES
        and value.ndim >= 1
        and value.stride(-1) == 1
        and residual.stride(-1) == 1
    )


def _torch_residual_layernorm(
    value: Tensor,
    residual: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> Tensor:
    summed = value.float() + residual.float()
    out = F.layer_norm(
        summed,
        normalized_shape,
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    )
    return out.to(value.dtype)


def _torch_residual_gelu(value: Tensor, residual: Tensor, approximate: str) -> Tensor:
    summed = value.float() + residual.float()
    return F.gelu(summed, approximate=approximate).to(value.dtype)


def fused_residual_layernorm(
    value: Tensor,
    residual: Tensor,
    normalized_shape: int | Sequence[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Return ``LayerNorm(value + residual)`` over ``normalized_shape``.

    The residual sum and the mean/variance statistics are computed in FP32;
    the result is cast back to ``value.dtype``. ``weight`` and ``bias`` follow
    :func:`torch.nn.functional.layer_norm` semantics and may be ``None``.

    The output preserves the shape, dtype, and device of ``value``.
    """
    _check_pair("fused_residual_layernorm", value, residual)
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(int(dim) for dim in normalized_shape)
    if len(normalized_shape) == 0 or any(dim <= 0 for dim in normalized_shape):
        raise ValueError(
            "fused_residual_layernorm: normalized_shape must be a non-empty "
            f"sequence of positive ints, got {normalized_shape}"
        )
    if (
        value.ndim < len(normalized_shape)
        or tuple(value.shape[-len(normalized_shape) :]) != normalized_shape
    ):
        raise ValueError(
            f"fused_residual_layernorm: normalized_shape {normalized_shape} does "
            f"not match the trailing dims of value {tuple(value.shape)}"
        )
    _check_affine("fused_residual_layernorm", "weight", weight, normalized_shape)
    _check_affine("fused_residual_layernorm", "bias", bias, normalized_shape)
    if weight is not None and weight.device != value.device:
        raise ValueError(
            f"fused_residual_layernorm: weight must be on {value.device}, got "
            f"{weight.device}"
        )
    if bias is not None and bias.device != value.device:
        raise ValueError(
            f"fused_residual_layernorm: bias must be on {value.device}, got "
            f"{bias.device}"
        )

    if value.numel() == 0 or not _triton_ready(value, residual):
        return _torch_residual_layernorm(
            value, residual, normalized_shape, weight, bias, eps
        )

    value_c = value.contiguous()
    residual_c = residual.contiguous()
    out = torch.empty_like(value_c)
    channels = 1
    for dim in normalized_shape:
        channels *= dim
    rows = value.numel() // channels
    block_c = min(1024, triton.next_power_of_2(channels))
    _residual_layernorm_kernel[(rows,)](
        value_c,
        residual_c,
        weight.contiguous() if weight is not None else value_c,
        bias.contiguous() if bias is not None else value_c,
        out,
        channels,
        eps,
        BLOCK_C=block_c,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return out


def fused_residual_gelu(
    value: Tensor,
    residual: Tensor,
    approximate: str = "none",
) -> Tensor:
    """Return ``GELU(value + residual)`` with FP32 accumulation.

    ``approximate`` follows :func:`torch.nn.functional.gelu`: ``"none"`` for
    the exact erf form (matching the FP8 GEMM epilogue) or ``"tanh"`` for the
    tanh approximation. The output preserves the shape, dtype, and device of
    ``value``.
    """
    _check_pair("fused_residual_gelu", value, residual)
    if approximate not in ("none", "tanh"):
        raise ValueError(
            f"fused_residual_gelu: approximate must be 'none' or 'tanh', got "
            f"{approximate!r}"
        )

    if value.numel() == 0 or not _triton_ready(value, residual):
        return _torch_residual_gelu(value, residual, approximate)

    value_c = value.contiguous()
    residual_c = residual.contiguous()
    out = torch.empty_like(value_c)
    numel = value.numel()
    block = 1024
    _residual_gelu_kernel[(triton.cdiv(numel, block),)](
        value_c,
        residual_c,
        out,
        numel,
        BLOCK=block,
        TANH=approximate == "tanh",
        num_warps=4,
    )
    return out


def _normalized_shape(
    op: str, normalized_shape: int | Sequence[int]
) -> tuple[int, ...]:
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(int(dim) for dim in normalized_shape)
    if len(normalized_shape) == 0 or any(dim <= 0 for dim in normalized_shape):
        raise ValueError(
            f"{op}: normalized_shape must be a non-empty sequence of positive "
            f"ints, got {normalized_shape}"
        )
    return normalized_shape


def _check_layernorm_value(
    op: str,
    value: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None,
    bias: Tensor | None,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{op}: value must be a torch.Tensor")
    if value.ndim < 2 or not value.is_floating_point():
        raise ValueError(
            f"{op}: value must be a floating tensor with at least 2 dims, got "
            f"shape {tuple(value.shape)} dtype {getattr(value, 'dtype', None)}"
        )
    if (
        value.ndim < len(normalized_shape)
        or tuple(value.shape[-len(normalized_shape) :]) != normalized_shape
    ):
        raise ValueError(
            f"{op}: normalized_shape {normalized_shape} does not match the "
            f"trailing dims of value {tuple(value.shape)}"
        )
    _check_affine(op, "weight", weight, normalized_shape)
    _check_affine(op, "bias", bias, normalized_shape)
    if weight is not None and weight.device != value.device:
        raise ValueError(f"{op}: weight must be on {value.device}, got {weight.device}")
    if bias is not None and bias.device != value.device:
        raise ValueError(f"{op}: bias must be on {value.device}, got {bias.device}")


_ROUND_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def _cuda_row_major(value: Tensor, channels: int) -> bool:
    return (
        triton is not None
        and value.is_cuda
        and value.dtype in _TRITON_DTYPES
        and value.is_contiguous()
        and value.numel() > 0
        and channels > 0
        and value.numel() % channels == 0
    )


def _scale_on_device(value: Tensor, scale_exponent: Tensor, padded: int) -> Tensor:
    scale = scale_exponent.to(device=value.device, dtype=torch.int8)
    expected = (padded // 32,)
    if tuple(scale.shape) != expected:
        raise ValueError(f"Expected scale shape {expected}, got {tuple(scale.shape)}")
    return scale


def _empty_quantized(value: Tensor, padded: int) -> Tensor:
    return torch.empty(
        (*value.shape[:-1], padded),
        device=value.device,
        dtype=torch.float8_e4m3fn,
    )


def _gelu_quantize_cuda(
    value: Tensor, scale_exponent: Tensor, approximate: str
) -> QuantizedTensor | None:
    channels = int(value.shape[-1])
    if not _cuda_row_major(value, channels):
        return None
    padded = ((channels + 31) // 32) * 32
    scale = _scale_on_device(value, scale_exponent, padded)
    out = _empty_quantized(value, padded)
    rows = value.numel() // channels
    block = 1024
    _gelu_quantize_kernel[(triton.cdiv(rows * padded, block),)](
        value,
        scale,
        out,
        channels,
        padded,
        rows * padded,
        BLOCK=block,
        TANH=approximate == "tanh",
        ROUND=_ROUND_CODE[value.dtype],
        num_warps=4,
    )
    return QuantizedTensor(out, scale, tuple(value.shape))


def _layernorm_quantize_cuda(
    value: Tensor,
    residual: Tensor | None,
    normalized_shape: tuple[int, ...],
    scale_exponent: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> QuantizedTensor | None:
    channels = 1
    for dim in normalized_shape:
        channels *= int(dim)
    if not _cuda_row_major(value, channels):
        return None
    if residual is not None and (
        residual.shape != value.shape or not _cuda_row_major(residual, channels)
    ):
        return None
    padded = ((channels + 31) // 32) * 32
    scale = _scale_on_device(value, scale_exponent, padded)
    out = _empty_quantized(value, padded)
    rows = value.numel() // channels
    block_c = min(1024, triton.next_power_of_2(channels))
    _layernorm_quantize_kernel[(rows,)](
        value,
        residual if residual is not None else value,
        weight.contiguous().reshape(-1) if weight is not None else value,
        bias.contiguous().reshape(-1) if bias is not None else value,
        scale,
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
    return QuantizedTensor(out, scale, tuple(value.shape))


def gelu_quantize_block32(
    value: Tensor,
    scale_exponent: Tensor,
    approximate: str = "none",
) -> QuantizedTensor:
    """Pack ``GELU(value)`` with a static block-32 scale.

    GELU runs in ``value.dtype`` (not an unrounded FP32 activation). The
    packed result is ``quantize_block32`` of that rounded tensor.
    """
    if not isinstance(value, Tensor):
        raise TypeError("gelu_quantize_block32: value must be a torch.Tensor")
    if value.ndim < 2 or not value.is_floating_point():
        raise ValueError(
            "gelu_quantize_block32: value must be a floating tensor with at "
            f"least 2 dims, got shape {tuple(value.shape)} dtype {value.dtype}"
        )
    if approximate not in ("none", "tanh"):
        raise ValueError(
            "gelu_quantize_block32: approximate must be 'none' or 'tanh', got "
            f"{approximate!r}"
        )
    packed = _gelu_quantize_cuda(value, scale_exponent, approximate)
    if packed is not None:
        return packed
    return quantize_block32(F.gelu(value, approximate=approximate), scale_exponent)


def layernorm_quantize_block32(
    value: Tensor,
    normalized_shape: int | Sequence[int],
    scale_exponent: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> QuantizedTensor:
    """Pack ``LayerNorm(value)`` with a static block-32 scale.

    Statistics are computed in FP32 and the dense result is cast back to
    ``value.dtype`` before quantization. ``weight`` and ``bias`` may be
    ``None``. The static scale is not recomputed from amax.
    """
    normalized_shape = _normalized_shape("layernorm_quantize_block32", normalized_shape)
    _check_layernorm_value(
        "layernorm_quantize_block32", value, normalized_shape, weight, bias
    )
    packed = _layernorm_quantize_cuda(
        value,
        None,
        normalized_shape,
        scale_exponent,
        weight,
        bias,
        eps,
    )
    if packed is not None:
        return packed
    normalized = F.layer_norm(
        value.float(),
        normalized_shape,
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).to(value.dtype)
    return quantize_block32(normalized, scale_exponent)


def residual_layernorm_quantize_block32(
    value: Tensor,
    residual: Tensor,
    normalized_shape: int | Sequence[int],
    scale_exponent: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> QuantizedTensor:
    """Pack ``LayerNorm(value + residual)`` with a static block-32 scale.

    The dense norm is :func:`fused_residual_layernorm` (Triton on contiguous
    CUDA fp16/bf16/fp32, otherwise the FP32 torch reference). Quantization
    consumes that rounded dense tensor, not the unrounded FP32 sum.
    """
    normalized_shape = _normalized_shape(
        "residual_layernorm_quantize_block32", normalized_shape
    )
    _check_pair("residual_layernorm_quantize_block32", value, residual)
    _check_layernorm_value(
        "residual_layernorm_quantize_block32",
        value,
        normalized_shape,
        weight,
        bias,
    )
    packed = _layernorm_quantize_cuda(
        value,
        residual,
        normalized_shape,
        scale_exponent,
        weight,
        bias,
        eps,
    )
    if packed is not None:
        return packed
    normalized = fused_residual_layernorm(
        value,
        residual,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )
    return quantize_block32(normalized, scale_exponent)
