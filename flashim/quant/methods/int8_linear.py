"""INT8 linears: per-output-channel int8 weights with one calibrated per-tensor activation scale (static) or one scale
per token computed at run time (dynamic)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.quant.formats.int8 import quantize_symmetric, safe_scale, weight_scales
from flashim.quant.tensor import Int8Tensor


class StaticInt8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` (or a 1x1 ``nn.Conv2d`` via :meth:`from_conv1x1`).

    Weights: symmetric int8, ``amax / 127`` per output row. Inputs: ``round(x / s)`` saturated
    to +-127 with a *static* calibrated ``s``. The GEMM accumulates in INT32; the epilogue
    applies ``s * w_scale[n]`` and the bias in FP32 and returns BF16 (cast to the input dtype).
    """

    def __init__(
        self, weight_int8: Tensor, weight_scale: Tensor, bias: Tensor | None, input_scale: Tensor
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight_int8.shape
        device = weight_int8.device
        self.register_buffer("weight_int8", weight_int8.contiguous())
        self.register_buffer("weight_scale", weight_scale.to(device=device, dtype=torch.float32))
        self.register_buffer("input_scale", safe_scale(input_scale.to(device)).reshape(()))
        self.register_buffer(
            "bias", None if bias is None else bias.detach().to(device=device, dtype=torch.bfloat16)
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, input_scale: Tensor) -> StaticInt8Linear:
        weight = linear.weight.detach()
        if not torch.isfinite(weight).all():
            raise ValueError("Cannot quantize non-finite weights")
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach()
            bias_mask = getattr(linear, "bias_mask", None)
            if bias_mask is not None:  # DINOv3 K-bias mask, folded once
                bias = bias * bias_mask.to(bias.dtype)
        scale = weight_scales(weight)
        return cls(quantize_symmetric(weight, scale[:, None]), scale, bias, input_scale)

    @classmethod
    def from_conv1x1(cls, conv: nn.Conv2d, input_scale: Tensor) -> StaticInt8Linear:
        if conv.kernel_size != (1, 1) or conv.groups != 1 or conv.stride != (1, 1) or conv.padding != (0, 0):
            raise ValueError("from_conv1x1 needs an ungrouped, unpadded, stride-1 1x1 convolution")
        linear = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
        linear.weight = nn.Parameter(conv.weight.detach()[:, :, 0, 0], requires_grad=False)
        if conv.bias is not None:
            linear.bias = nn.Parameter(conv.bias.detach(), requires_grad=False)
        return cls.from_linear(linear, input_scale)

    def quantize(self, x: Tensor) -> Int8Tensor:
        return ops.quantize_int8(x, self.input_scale)

    def quantize_layernorm(
        self, value: Tensor, residual: Tensor | None, norm: nn.LayerNorm, *, return_sum: bool = False,
        activation: str | None = None,
    ) -> Int8Tensor | tuple[Int8Tensor, Tensor]:
        """``norm(value + residual)`` packed for this layer (``activation``: then GELU); ``return_sum`` also returns
        ``value + residual``."""
        return ops.layernorm_quantize_int8(
            value, residual, norm.weight, norm.bias, norm.eps, self.input_scale, return_sum=return_sum,
            activation=activation,
        )

    def quantize_gelu(self, value: Tensor, approximate: str) -> Int8Tensor:
        return ops.gelu_quantize_int8(value, self.input_scale, approximate)

    def forward_prequantized(self, quantized: Int8Tensor, relu: bool = False) -> Tensor:
        """Consume an activation packed with this layer's input scale; BF16 out (``relu``: ReLU in the epilogue)."""
        if quantized.scale is not self.input_scale and not torch.equal(quantized.scale, self.input_scale):
            raise ValueError("Prequantized activation uses a different static scale")
        out = ops.int8_linear(
            quantized.matrix(), self.weight_int8, self.input_scale, self.weight_scale, self.bias, relu=relu
        )
        return out.reshape(*quantized.logical_shape[:-1], self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_prequantized(self.quantize(x)).to(x.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8"


class StaticInt8Conv1x1(StaticInt8Linear):
    """A 1x1 convolution on ``[B, C, H, W]`` as an INT8 GEMM over the ``B*H*W`` pixels (BF16 out); ``forward(x,
    relu=True)`` applies a following ReLU in the GEMM epilogue (:func:`flashim.layers.conv.conv_relu`)."""

    fuses_relu = True

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, input_scale: Tensor) -> StaticInt8Conv1x1:
        inner = StaticInt8Linear.from_conv1x1(conv, input_scale)
        return cls(inner.weight_int8, inner.weight_scale, inner.bias, inner.input_scale)

    def forward(self, x: Tensor, relu: bool = False) -> Tensor:
        pixels = x.permute(0, 2, 3, 1)  # channels-last view: contiguous when x is channels-last
        if not (pixels.is_contiguous() or x.is_contiguous()):  # NCHW memory is read in place by the quantize kernel
            pixels = pixels.contiguous()
        out = self.forward_prequantized(self.quantize(pixels), relu=relu).permute(0, 3, 1, 2)
        return out.to(x.dtype)


class DynamicInt8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with per-token (per-row) dynamic INT8 activations: each row is quantized
    with its own ``amax / 127`` at run time (``ops.quantize_int8_rows``) and the INT8 GEMM applies the row scale in its
    epilogue; weights as :class:`StaticInt8Linear`. For activations whose outliers are whole tokens (DINOv2's
    "massive activations"), which one static per-tensor scale cannot represent; needs no calibration."""

    def __init__(self, weight_int8: Tensor, weight_scale: Tensor, bias: Tensor | None) -> None:
        super().__init__()
        self.out_features, self.in_features = weight_int8.shape
        device = weight_int8.device
        self.register_buffer("weight_int8", weight_int8.contiguous())
        self.register_buffer("weight_scale", weight_scale.to(device=device, dtype=torch.float32))
        self.register_buffer(
            "bias", None if bias is None else bias.detach().to(device=device, dtype=torch.bfloat16)
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> DynamicInt8Linear:
        static = StaticInt8Linear.from_linear(linear, torch.ones((), device=linear.weight.device))
        return cls(static.weight_int8, static.weight_scale, static.bias)

    def forward(self, x: Tensor) -> Tensor:
        rows = x.reshape(-1, self.in_features)
        values, scale = ops.quantize_int8_rows(rows if rows.is_contiguous() else rows.contiguous())
        out = ops.int8_linear(values, self.weight_int8, scale, self.weight_scale, self.bias)
        return out.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8, dynamic per-token"
