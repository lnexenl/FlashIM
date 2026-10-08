"""GEMMs through torch's own vendor-library bindings (cuBLAS / cuBLASLt)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature
from flashim.quant.formats import mx
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight


@torch.compiler.allow_in_graph
def _scaled_mm_cublas(a: Tensor, b_t: Tensor, a_scale: Tensor, b_scale: Tensor, bias: Tensor | None) -> Tensor:
    # The ScalingType/SwizzleType pybind enums cannot be proxied by Dynamo, so they
    # stay inside this allow_in_graph boundary.
    return F.scaled_mm(
        a,
        b_t,
        a_scale,
        F.ScalingType.BlockWise1x32,
        b_scale,
        F.ScalingType.BlockWise1x32,
        F.SwizzleType.SWIZZLE_32_4_4,
        F.SwizzleType.SWIZZLE_32_4_4,
        bias=bias,
        output_dtype=torch.bfloat16,
    )


@ops.mxfp8_scaled_mm.register("cublas", device=DeviceSpec("cuda"), requires=[Feature.MXFP8_SCALED_MM], priority=1)
def _mxfp8_scaled_mm_cublas(a, b_t, a_scale, b_scale, bias=None):
    return _scaled_mm_cublas(a, b_t, a_scale, b_scale, bias)


class _InputScales:
    """Swizzled static input scales per row count for one layer's calibrated input exponent (the first one seen;
    another exponent object is converted per call, uncached)."""

    def __init__(self) -> None:
        self.exponent: Tensor | None = None
        self.by_rows: dict[int, Tensor] = {}

    def get(self, exponent: Tensor, rows: int) -> Tensor:
        if self.exponent is None:
            self.exponent = exponent
        if exponent is not self.exponent:
            return mx.static_input_scales(exponent, rows)
        scales = self.by_rows.get(rows)
        if scales is None:
            scales = self.by_rows[rows] = mx.static_input_scales(exponent, rows)
        return scales


def _prepare_mxfp8(weight: QuantizedWeight) -> tuple[Tensor, Tensor, _InputScales]:
    return weight.values.t(), mx.swizzle_scales(weight.scale_exponent), _InputScales()


def _cublas_shapes(weight: QuantizedWeight) -> bool:
    # cuBLAS block-scaled MXFP8: padded N and K multiples of 128.
    return weight.out_features % 128 == 0 and weight.in_features % 128 == 0


@ops.mxfp8_linear.register(
    "cublas", device=DeviceSpec("cuda"), requires=[Feature.MXFP8_SCALED_MM], priority=1,
    prepare=_prepare_mxfp8, accepts=_cublas_shapes,
)
def _mxfp8_linear_cublas(
    activation: QuantizedTensor, bias: Tensor | None, weight_t: Tensor, weight_scale: Tensor, input_scales: _InputScales
) -> Tensor:
    matrix = activation.matrix()
    a_scale = input_scales.get(activation.scale_exponent, matrix.shape[0])
    return _scaled_mm_cublas(
        matrix, weight_t, a_scale, weight_scale, None if bias is None else bias.to(torch.bfloat16)
    )


def _int_mm_eligible(a, weight, a_scale, w_scale, bias, **_) -> bool:
    # torch._int_mm: M > 16, K and N multiples of 8 (cuBLASLt IMMA).
    return a.shape[0] > 16 and a.shape[1] % 8 == 0 and weight.shape[0] % 8 == 0 and a.is_contiguous()


@ops.int8_linear.register("cublas", device=DeviceSpec("cuda", min_capability=(7, 5)), when=_int_mm_eligible)
def _int8_linear_cublas(a, weight, a_scale, w_scale, bias, *, relu: bool = False):
    out = torch._int_mm(a, weight.t()).float() * (a_scale * w_scale)
    if bias is not None:
        out = out + bias.float()
    if relu:
        out = out.relu()
    return out.to(torch.bfloat16)
