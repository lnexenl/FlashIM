"""Fused producer -> static MXFP8 quantization: the norm/activation feeding a quantized
linear writes its E4M3 input directly."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op
from flashim.platforms import Feature
from flashim.quant.tensor import Int8Tensor, QuantizedTensor, quantize_int8_static, quantize_static

_TRITON_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


def _cuda_row_major(value: Tensor) -> bool:
    return value.dtype in _TRITON_FLOATS and value.is_contiguous() and value.numel() > 0


# -- LayerNorm (+ residual) -> static quantize ------------------------------------------


def _layernorm_quantize_reference(
    value: Tensor,
    residual: Tensor | None,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
    scale_exponent: Tensor,
    *,
    input_bias: Tensor | None = None,
    return_dense: bool = False,
) -> QuantizedTensor | tuple[QuantizedTensor, Tensor]:
    """``quantize_static(LayerNorm(value [+ input_bias] [+ residual]))`` with FP32 statistics.

    The normalized tensor is rounded to ``value.dtype`` before quantization. ``return_dense`` also returns
    that rounded tensor: ``(quantized, dense)``.
    """
    dense = value.float() if input_bias is None else value.float() + input_bias.float()
    dense = dense if residual is None else dense + residual.float()
    normalized = F.layer_norm(
        dense,
        (value.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).to(value.dtype)
    quantized = quantize_static(normalized, scale_exponent)
    return (quantized, normalized) if return_dense else quantized


layernorm_quantize = Op("layernorm_quantize", _layernorm_quantize_reference)


def _layernorm_quantize_eligible(value, residual, weight, bias, eps, scale_exponent, **kwargs) -> bool:
    return _cuda_row_major(value) and (
        residual is None or (residual.shape == value.shape and _cuda_row_major(residual))
    )


@layernorm_quantize.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_layernorm_quantize_eligible
)
def _layernorm_quantize_triton(value, residual, weight, bias, eps, scale_exponent, **kwargs):
    from flashim.ops._triton.quantize import layernorm_quantize as kernel

    return kernel(value, residual, weight, bias, eps, scale_exponent.to(value.device, torch.int8), **kwargs)


# -- GELU -> static quantize ---------------------------------------------------------------


def _gelu_quantize_reference(
    value: Tensor, scale_exponent: Tensor, approximate: str = "none", *, input_bias: Tensor | None = None
) -> QuantizedTensor:
    """``quantize_static(GELU(value [+ input_bias]))``; GELU is evaluated in ``value.dtype`` (FP32 when a bias is
    fused)."""
    if input_bias is not None:
        activated = F.gelu(value.float() + input_bias.float(), approximate=approximate).to(value.dtype)
        return quantize_static(activated, scale_exponent)
    return quantize_static(F.gelu(value, approximate=approximate), scale_exponent)


gelu_quantize = Op("gelu_quantize", _gelu_quantize_reference)


@gelu_quantize.register(
    "triton",
    device=DeviceSpec("cuda"),
    requires=[Feature.TRITON],
    when=lambda value, *args, **kwargs: _cuda_row_major(value),
)
def _gelu_quantize_triton(value: Tensor, scale_exponent: Tensor, approximate: str = "none", *, input_bias=None):
    from flashim.ops._triton.quantize import gelu_quantize as kernel

    return kernel(value, scale_exponent.to(value.device, torch.int8), approximate, input_bias=input_bias)


# -- plain static quantize ---------------------------------------------------------------------------


def _quantize_static_reference(value: Tensor, scale_exponent: Tensor) -> QuantizedTensor:
    """Static block-32 E4M3 quantization of a dense activation (pads channels to a multiple of 32)."""
    return quantize_static(value, scale_exponent)


quantize_mxfp8 = Op("quantize_mxfp8", _quantize_static_reference)


@quantize_mxfp8.register(
    "triton",
    device=DeviceSpec("cuda"),
    requires=[Feature.TRITON],
    when=lambda value, scale_exponent: _cuda_row_major(value),
)
def _quantize_static_triton(value: Tensor, scale_exponent: Tensor) -> QuantizedTensor:
    from flashim.ops._triton.quantize import quantize_static as kernel

    return kernel(value, scale_exponent.to(value.device, torch.int8))


# -- INT8 producers: static per-tensor FP32 scale ---------------------------------------------------


def _quantize_int8_reference(value: Tensor, scale: Tensor) -> Int8Tensor:
    """``round(value / scale)`` saturated to +-127; ``scale`` is a single FP32 value."""
    return quantize_int8_static(value, scale)


quantize_int8 = Op("quantize_int8", _quantize_int8_reference)


@quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=lambda value, scale: _cuda_row_major(value)
)
def _quantize_int8_triton(value: Tensor, scale: Tensor) -> Int8Tensor:
    from flashim.ops._triton.quantize import quantize_int8 as kernel

    return kernel(value, scale)


def _layernorm_quantize_int8_reference(
    value: Tensor, residual: Tensor | None, weight: Tensor | None, bias: Tensor | None, eps: float, scale: Tensor
) -> Int8Tensor:
    """``quantize_int8(LayerNorm(value [+ residual]))`` with FP32 statistics, rounded to ``value.dtype`` first."""
    dense = value.float() if residual is None else value.float() + residual.float()
    normalized = F.layer_norm(
        dense,
        (value.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).to(value.dtype)
    return quantize_int8_static(normalized, scale)


layernorm_quantize_int8 = Op("layernorm_quantize_int8", _layernorm_quantize_int8_reference, composite=True)


@layernorm_quantize_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_layernorm_quantize_eligible
)
def _layernorm_quantize_int8_triton(value, residual, weight, bias, eps, scale):
    from flashim.ops._triton.quantize import layernorm_quantize_int8 as kernel

    return kernel(value, residual, weight, bias, eps, scale)


def _gelu_quantize_int8_reference(value: Tensor, scale: Tensor, approximate: str = "none") -> Int8Tensor:
    """``quantize_int8(GELU(value))``; GELU is evaluated in ``value.dtype``."""
    return quantize_int8_static(F.gelu(value, approximate=approximate), scale)


gelu_quantize_int8 = Op("gelu_quantize_int8", _gelu_quantize_int8_reference, composite=True)


@gelu_quantize_int8.register(
    "triton",
    device=DeviceSpec("cuda"),
    requires=[Feature.TRITON],
    when=lambda value, *args, **kwargs: _cuda_row_major(value),
)
def _gelu_quantize_int8_triton(value: Tensor, scale: Tensor, approximate: str = "none"):
    from flashim.ops._triton.quantize import gelu_quantize_int8 as kernel

    return kernel(value, scale, approximate)
