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


def _matmul_nt_fp32_eligible(a: Tensor, b: Tensor, scale: float) -> bool:
    return (
        a.ndim == b.ndim == 2 and a.dtype == b.dtype and a.dtype in (torch.bfloat16, torch.float16)
        and a.shape[1] == b.shape[1]
    )


# BF16 tensor-core GEMM, FP32 accumulation and output, the scale as cuBLAS' alpha (beta = 0: the scalar input is never
# read). LoMa's 4096^2 similarity, in situ (CUDA graph), dim 256 / 512 / 1024: L20 0.27 / 0.56 / 1.14 ms as an FP32
# SGEMM -> 0.09 / 0.17 / 0.32 ms (mm + a separate scale: 0.12 / 0.20 / 0.35; TF32: 0.18 / 0.37 / 0.72), H20 0.40 / 0.66 /
# 1.26 -> 0.08 / 0.13 / 0.27 ms; the best Triton tile (scale in the epilogue): 5-10% slower on L20, within 5% on H20.
# A custom op: torch 2.11's meta kernel of ``aten.addmm.dtype`` is broken ("multiple values for argument 'beta'"), so
# Dynamo cannot trace the call itself.
@torch.library.custom_op("flashim::matmul_nt_fp32_cublas", mutates_args=())
def _matmul_nt_fp32_addmm(a: Tensor, b: Tensor, scale: float) -> Tensor:
    zero = a.new_zeros((), dtype=torch.float32)
    return torch.addmm(zero, a, b.t(), beta=0, alpha=scale, out_dtype=torch.float32)


@_matmul_nt_fp32_addmm.register_fake
def _(a: Tensor, b: Tensor, scale: float) -> Tensor:
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.float32)


@ops.matmul_nt_fp32.register("cublas", device=DeviceSpec("cuda", min_capability=(8, 0)), when=_matmul_nt_fp32_eligible)
def _matmul_nt_fp32_cublas(a: Tensor, b: Tensor, scale: float) -> Tensor:
    return _matmul_nt_fp32_addmm(a, b, scale)
