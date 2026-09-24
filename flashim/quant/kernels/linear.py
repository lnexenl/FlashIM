"""Linear kernels for static MXFP8 layers (vLLM-style: ``is_supported`` + ``can_implement``).

Candidates are tried in priority order; the emulation kernel always applies, so a
quantized model runs on any platform (slowly) with the same numerics contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from flashim import ops
from flashim.platforms import Feature, Platform
from flashim.quant.formats import mx
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight


class MXFP8LinearKernel(ABC):
    name: str

    @classmethod
    @abstractmethod
    def is_supported(cls, platform: Platform) -> tuple[bool, str | None]: ...

    @classmethod
    def can_implement(cls, out_features: int, in_features: int) -> tuple[bool, str | None]:
        return True, None

    def prepare(self, weight: QuantizedWeight) -> dict[str, Tensor]:
        """Extra device buffers derived from the packed weight (built once at load)."""
        return {}

    @abstractmethod
    def apply(self, layer, activation: QuantizedTensor) -> Tensor:
        """Return the padded ``[rows, N]`` BF16 product (bias not added)."""


class CublasMXFP8Kernel(MXFP8LinearKernel):
    """cuBLAS block-scaled GEMM via ``F.scaled_mm`` (Blackwell); padded K, N % 128 == 0."""

    name = "cublas_mxfp8"

    @classmethod
    def is_supported(cls, platform):
        ok = platform.supports(Feature.MXFP8_SCALED_MM)
        return ok, None if ok else "no block-scaled MXFP8 scaled_mm"

    @classmethod
    def can_implement(cls, out_features, in_features):
        ok = out_features % 128 == 0 and in_features % 128 == 0
        return ok, None if ok else f"padded N={out_features}, K={in_features} not multiples of 128"

    def prepare(self, weight):
        return {"weight_scale_swizzled": mx.swizzle_scales(weight.scale_exponent)}

    def apply(self, layer, activation):
        matrix = activation.matrix()
        return ops.mxfp8_scaled_mm.get("cublas")(
            matrix,
            layer.weight_values.view(activation.values.dtype).t(),
            layer.input_scales(matrix.shape[0]),
            layer.weight_scale_swizzled,
            None,
        )


class TritonFP8Kernel(MXFP8LinearKernel):
    """Triton FP8 GEMM (``tl.dot`` on E4M3) for shapes cuBLAS MXFP8 cannot take."""

    name = "triton_fp8"

    @classmethod
    def is_supported(cls, platform):
        ok = platform.supports(Feature.TRITON, Feature.FP8_E4M3)
        return ok, None if ok else "needs Triton + FP8 (sm_89+)"

    def apply(self, layer, activation):
        return ops.fp8_linear.get("triton")(activation, layer.packed_weight(), None)


class EmulationKernel(MXFP8LinearKernel):
    """Dequantize-and-matmul reference; runs anywhere."""

    name = "emulation"

    @classmethod
    def is_supported(cls, platform):
        return True, None

    def apply(self, layer, activation):
        return ops.fp8_linear.get("reference")(activation, layer.packed_weight(), None)


KERNELS: tuple[type[MXFP8LinearKernel], ...] = (CublasMXFP8Kernel, TritonFP8Kernel, EmulationKernel)


def choose_kernel(platform: Platform, out_features: int, in_features: int) -> MXFP8LinearKernel:
    reasons = []
    for kernel in KERNELS:
        supported, why = kernel.is_supported(platform)
        if supported:
            implementable, why = kernel.can_implement(out_features, in_features)
            if implementable:
                return kernel()
        reasons.append(f"{kernel.name}: {why}")
    raise RuntimeError("No MXFP8 linear kernel: " + "; ".join(reasons))
