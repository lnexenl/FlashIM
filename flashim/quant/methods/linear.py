"""Static MXFP8 linear: packed weights + calibrated block-32 activation scales."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.ops import BoundOp
from flashim.quant.formats import mx
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight, quantize_static


class StaticMXFP8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear``.

    Weights: E4M3 with one E8M0 exponent per (output row, 32 inputs), padded to
    multiples of 32. Inputs: quantized with a *static* calibrated exponent per
    32-channel block (values beyond the calibrated range saturate). Output: BF16
    GEMM result plus BF16 bias, cast to the input dtype (RoMaV2 FastFP8 contract).

    The GEMM is ``ops.mxfp8_linear`` bound to this layer's packed weight: cuBLAS block-scaled MXFP8 where it
    applies (Blackwell, N and K multiples of 128), else the Triton FP8 GEMM, else the emulated reference. ``impl``
    pins one (e.g. ``"reference"``). Moving the layer (``.to()``) rebinds it.
    """

    def __init__(
        self,
        weight: QuantizedWeight,
        bias: Tensor | None,
        input_exponent: Tensor,
        *,
        impl: str | None = None,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = weight.logical_shape[1], weight.logical_shape[0]
        self.impl = impl
        device = weight.values.device
        self.register_buffer("weight_values", weight.values.view(torch.uint8))
        self.register_buffer("weight_exponent", weight.scale_exponent)
        self.register_buffer("input_exponent", input_exponent.to(device=device, dtype=torch.int8))
        self.register_buffer(
            "bias", None if bias is None else F.pad(bias.float(), (0, weight.out_features - bias.shape[0]))
        )
        self.register_buffer("bias_bf16", None if self.bias is None else self.bias.to(torch.bfloat16))
        self._bind()

    def _bind(self) -> None:
        self.gemm: BoundOp = ops.mxfp8_linear.bind(self.packed_weight(), impl=self.impl).warmup()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        self._bind()  # the bound weights (and their packed forms) must follow the buffers
        return result

    @property
    def kernel_name(self) -> str:
        """The GEMM impl calls will use (``"cublas"``, ``"triton"`` or ``"reference"``)."""
        return self.gemm.primary

    @classmethod
    def from_linear(cls, linear: nn.Linear, input_exponent: Tensor, *, impl: str | None = None) -> StaticMXFP8Linear:
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
        return cls(weight, bias, input_exponent, impl=impl)

    def packed_weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            self.weight_values.view(torch.float8_e4m3fn),
            self.weight_exponent,
            (self.out_features, self.in_features),
        )

    def _finish(self, out: Tensor, leading: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        if self.bias_bf16 is not None:
            out = out + self.bias_bf16
        return out[:, : self.out_features].reshape(*leading, self.out_features).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        quantized = quantize_static(x, self.input_exponent)
        return self._finish(self.gemm(quantized, None), tuple(x.shape[:-1]), x.dtype)

    def quantize_layernorm(self, value: Tensor, residual: Tensor | None, norm: nn.LayerNorm) -> QuantizedTensor:
        return ops.layernorm_quantize(value, residual, norm.weight, norm.bias, norm.eps, self.input_exponent)

    def quantize_gelu(self, value: Tensor, approximate: str) -> QuantizedTensor:
        return ops.gelu_quantize(value, self.input_exponent, approximate)

    def matmul(self, quantized: QuantizedTensor, *, bias: bool = False) -> Tensor:
        """BF16 product of an activation packed with this layer's input exponent: ``[*leading, out_features]``.

        ``bias=False`` leaves the bias out (for consumers that fold it into their own kernel); ``bias=True`` adds
        it in the GEMM epilogue (FP32 accumulator + bias, one rounding)."""
        out = self.gemm(quantized, self.bias if bias else None)
        return out[:, : self.out_features].reshape(*quantized.logical_shape[:-1], self.out_features)

    def forward_prequantized(self, quantized: QuantizedTensor) -> Tensor:
        """Consume an activation already packed with this layer's input exponent; BF16 out."""
        if quantized.scale_exponent is not self.input_exponent and not torch.equal(
            quantized.scale_exponent, self.input_exponent
        ):
            raise ValueError("Prequantized activation uses a different static scale")
        out = self.gemm(quantized, None)
        return self._finish(out, quantized.logical_shape[:-1], torch.bfloat16)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, kernel={self.kernel_name}"
