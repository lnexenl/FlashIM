"""Static MXFP8 linear: packed weights + calibrated block-32 activation scales."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim.platforms import Platform
from flashim.quant.formats import mx
from flashim.quant.kernels.linear import MXFP8LinearKernel, choose_kernel
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight, quantize_static


class StaticMXFP8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear``.

    Weights: E4M3 with one E8M0 exponent per (output row, 32 inputs), padded to
    multiples of 32. Inputs: quantized with a *static* calibrated exponent per
    32-channel block (values beyond the calibrated range saturate). Output: BF16
    GEMM result plus BF16 bias, cast to the input dtype (RoMaV2 FastFP8 contract).
    """

    def __init__(
        self,
        weight: QuantizedWeight,
        bias: Tensor | None,
        input_exponent: Tensor,
        *,
        kernel: MXFP8LinearKernel,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = weight.logical_shape[1], weight.logical_shape[0]
        self.kernel = kernel
        device = weight.values.device
        self.register_buffer("weight_values", weight.values.view(torch.uint8))
        self.register_buffer("weight_exponent", weight.scale_exponent)
        self.register_buffer("input_exponent", input_exponent.to(device=device, dtype=torch.int8))
        self.register_buffer(
            "bias", None if bias is None else F.pad(bias.float(), (0, weight.out_features - bias.shape[0]))
        )
        self.register_buffer("bias_bf16", None if self.bias is None else self.bias.to(torch.bfloat16))
        for name, value in kernel.prepare(weight).items():
            self.register_buffer(name, value)
        self._input_scales: dict[int, Tensor] = {}
        self.call_count = 0

    @classmethod
    def from_linear(cls, linear: nn.Linear, input_exponent: Tensor, platform: Platform) -> StaticMXFP8Linear:
        weight = QuantizedWeight.from_float(linear.weight.detach())
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach()
            bias_mask = getattr(linear, "bias_mask", None)
            if bias_mask is not None:  # DINOv3 K-bias mask, folded once
                if not torch.isfinite(bias_mask).all():
                    raise ValueError("bias_mask contains non-finite values")
                bias = bias * bias_mask.to(bias.dtype)
        expected = (weight.in_features // mx.BLOCK,)
        if tuple(input_exponent.shape) != expected:
            raise ValueError(f"Input exponent shape {tuple(input_exponent.shape)} != {expected}")
        kernel = choose_kernel(platform, weight.out_features, weight.in_features)
        return cls(weight, bias, input_exponent, kernel=kernel)

    def packed_weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            self.weight_values.view(torch.float8_e4m3fn),
            self.weight_exponent,
            (self.out_features, self.in_features),
        )

    def input_scales(self, rows: int) -> Tensor:
        """Swizzled static input scales for ``rows`` rows (cached per row count)."""
        scales = self._input_scales.get(rows)
        if scales is None:
            scales = mx.static_input_scales(self.input_exponent, rows)
            self._input_scales[rows] = scales
        return scales

    def _finish(self, out: Tensor, leading: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        if self.bias_bf16 is not None:
            out = out + self.bias_bf16
        return out[:, : self.out_features].reshape(*leading, self.out_features).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        if not torch.compiler.is_compiling():
            self.call_count += 1
        quantized = quantize_static(x, self.input_exponent)
        return self._finish(self.kernel.apply(self, quantized), tuple(x.shape[:-1]), x.dtype)

    def forward_prequantized(self, quantized: QuantizedTensor) -> Tensor:
        """Consume an activation already packed with this layer's input exponent; BF16 out."""
        if not torch.compiler.is_compiling():
            self.call_count += 1
        if quantized.scale_exponent is not self.input_exponent and not torch.equal(
            quantized.scale_exponent, self.input_exponent
        ):
            raise ValueError("Prequantized activation uses a different static scale")
        out = self.kernel.apply(self, quantized)
        return self._finish(out, quantized.logical_shape[:-1], torch.bfloat16)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, kernel={self.kernel.name}"
