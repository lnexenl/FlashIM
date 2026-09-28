"""Fused producer -> static MXFP8 quantization: the norm/activation feeding a quantized
linear writes its E4M3 input directly."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.quant.tensor import Int8Tensor, QuantizedTensor, quantize_int8_static, quantize_static

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


# -- plain static quantize ---------------------------------------------------------------------------


def _quantize_static_reference(value: Tensor, scale_exponent: Tensor) -> QuantizedTensor:
    """Static block-32 E4M3 quantization of a dense activation (pads channels to a multiple of 32)."""
    return quantize_static(value, scale_exponent)


quantize_mxfp8 = Op("quantize_mxfp8", _quantize_static_reference)


# -- INT8 producers: static per-tensor FP32 scale ---------------------------------------------------


def _quantize_int8_reference(value: Tensor, scale: Tensor) -> Int8Tensor:
    """``round(value / scale)`` saturated to +-127; ``scale`` is a single FP32 value."""
    return quantize_int8_static(value, scale)


quantize_int8 = Op("quantize_int8", _quantize_int8_reference)


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


layernorm_quantize_int8 = Op("layernorm_quantize_int8", _layernorm_quantize_int8_reference)


def _gelu_quantize_int8_reference(value: Tensor, scale: Tensor, approximate: str = "none") -> Int8Tensor:
    """``quantize_int8(GELU(value))``; GELU is evaluated in ``value.dtype``."""
    return quantize_int8_static(F.gelu(value, approximate=approximate), scale)


gelu_quantize_int8 = Op("gelu_quantize_int8", _gelu_quantize_int8_reference)
