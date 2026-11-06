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


def _int_mm_eligible(a, weight, a_scale, w_scale, bias, *, gelu=False, residual=None, **_) -> bool:
    # torch._int_mm: M > 16, K and N multiples of 8 (cuBLASLt IMMA); no GELU / residual epilogue.
    return (
        a.shape[0] > 16 and a.shape[1] % 8 == 0 and weight.shape[0] % 8 == 0 and a.is_contiguous()
        and not gelu and residual is None
    )


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


# -- cuBLAS blockwise FP8 (1x128 activations x 128x128 weights) on Hopper ---------------------------------------
#
# ``F.scaled_mm`` with ``BlockWise1x128`` (A) x ``BlockWise128x128`` (B) runs the FlexGlue linears 2.2x (B=1) to 2.4x
# (B=16) faster than the Triton block-32 kernel on an H800 (81 launches: 987 us vs 2209 us per decode_pair, isolated),
# but its scales are coarser than FlashIM's block-32 ones: one per 128 activation channels, one per 128x128 weight
# block. A layer opts in by carrying ``QuantizedWeight.activation_block == 128`` (``StaticMXFP8Linear.
# set_activation_block``): its static input exponent is then constant over each group of 4 blocks, so the producers
# (``quantize_mxfp8`` / ``layernorm_quantize`` / ``gelu_quantize``, unchanged) already emit the cuBLAS payload and the
# GEMM needs no pass over the activation. Weights are re-encoded once in ``prepare``: ``q' = e4m3(q * 2^(e - e_block))``,
# ``e_block`` the max exponent of the 128x128 block; the scales are powers of two, so this is exact except where a value
# falls into E4M3's subnormal range (~0.01% of the weights).
# Layouts cuBLAS accepts (checked against an fp32 matmul of the dequantized operands, rel. error 1.66e-3 = the BF16
# output rounding): ``scale_a`` = the [rows, K/128] factors stored as [K/128, rows] (strides (1, rows)), ``scale_b`` =
# [N/128, K/128] row-major viewed as its transpose (shape [K/128, N/128], strides (1, K/128)); rows % 4 == 0.
@torch.library.custom_op("flashim::fp8_block128_mm", mutates_args=())
def _fp8_block128_mm(a: Tensor, w: Tensor, a_scale: Tensor, w_scale: Tensor) -> Tensor:
    """``a [M, K]`` E4M3 x ``w [N, K]`` E4M3^T -> BF16 ``[M, N]``; ``a_scale [K/128, M]`` and ``w_scale [N/128, K/128]``
    FP32, both contiguous (transposed into the strides cuBLAS wants here)."""
    return F.scaled_mm(
        a, w.t(), [a_scale.t()], [F.ScalingType.BlockWise1x128], [w_scale.t()], [F.ScalingType.BlockWise128x128],
        output_dtype=torch.bfloat16,
    )


@_fp8_block128_mm.register_fake
def _(a: Tensor, w: Tensor, a_scale: Tensor, w_scale: Tensor) -> Tensor:
    return a.new_empty((a.shape[0], w.shape[0]), dtype=torch.bfloat16)


def _group_scale(exponent: Tensor) -> Tensor:
    """FP32 ``2^e`` of each group of 4 block exponents (equal inside a group), ``[K/128]``; the all-zero marker -127
    gives scale 1 (its payload is 0)."""
    group = exponent[::4].to(torch.int16)
    return torch.exp2(torch.where(group == -127, torch.zeros_like(group), group).float())


class _BlockActivation:
    """Per-layer activation scale: the first exponent seen, and per row count the cuBLAS tensor ``[K/128, rows]``.
    Under ``torch.compile`` it is recomputed in the graph (a few elements, fused): a cache filled while tracing
    would hold fake tensors."""

    def __init__(self) -> None:
        self.exponent: Tensor | None = None
        self.group_scale: Tensor | None = None
        self.by_rows: dict[int, Tensor] = {}

    def get(self, exponent: Tensor, rows: int) -> Tensor:
        if torch.compiler.is_compiling():
            return _group_scale(exponent)[:, None].expand(-1, rows).contiguous()
        if self.exponent is None:
            self.exponent = exponent
            self.group_scale = _group_scale(exponent)
        if exponent is not self.exponent:  # another exponent object (not the layer's): converted per call, uncached
            return _group_scale(exponent)[:, None].expand(-1, rows).contiguous()
        scale = self.by_rows.get(rows)
        if scale is None:
            scale = self.by_rows[rows] = self.group_scale[:, None].expand(-1, rows).contiguous()
        return scale


def _prepare_block128(weight: QuantizedWeight) -> tuple[Tensor, Tensor, _BlockActivation]:
    """Re-encode the block-32 weight to 128x128 block scales, once: ``(values [N, K] E4M3, scales [N/128, K/128]
    FP32, the layer's activation state)``."""
    n, k = weight.values.shape
    e = weight.scale_exponent.to(torch.int16).reshape(n // 128, 128, k // 128, 4)
    block = e.amax(dim=(1, 3))
    block = torch.where(block == -127, torch.zeros_like(block), block)  # all-zero block: scale 1, payload stays 0
    shift = torch.exp2((e - block[:, None, :, None]).float())  # <= 1: only ever shrinks the payload
    values = weight.values.reshape(n // 128, 128, k // 128, 4, 32).float() * shift[..., None]
    return values.to(torch.float8_e4m3fn).reshape(n, k), torch.exp2(block.float()).contiguous(), _BlockActivation()


def _block128_weights(weight: QuantizedWeight) -> bool:
    # the layer asked for it (its input exponent is constant over 128 channels) and the shapes tile
    return weight.activation_block == 128 and weight.out_features % 128 == 0 and weight.in_features % 128 == 0


def _block128_eligible(activation: QuantizedTensor, bias: Tensor | None, weight: QuantizedWeight) -> bool:
    # rows % 4 == 0 (16-byte aligned scale_a); no bias epilogue. Other calls take the next impl (Triton).
    return bias is None and activation.rows % 4 == 0


@ops.mxfp8_linear.register(
    "cublas-block128", device=DeviceSpec("cuda", min_capability=(9, 0), max_capability=(9, 0)),
    requires=[Feature.FP8_E4M3], priority=2, prepare=_prepare_block128, accepts=_block128_weights,
    when=_block128_eligible,
)
def _mxfp8_linear_block128(
    activation: QuantizedTensor, bias: Tensor | None, values: Tensor, weight_scale: Tensor, state: _BlockActivation
) -> Tensor:
    a = activation.matrix()
    return _fp8_block128_mm(a, values, state.get(activation.scale_exponent, a.shape[0]), weight_scale)
