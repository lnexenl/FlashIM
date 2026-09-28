"""GEMMs through torch's own vendor-library bindings (cuBLAS / cuBLASLt)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


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


@ops.mxfp8_scaled_mm.register("cublas", device=DeviceSpec("cuda"), requires=[Feature.MXFP8_SCALED_MM])
def _mxfp8_scaled_mm_cublas(a, b_t, a_scale, b_scale, bias=None):
    return _scaled_mm_cublas(a, b_t, a_scale, b_scale, bias)


def _int_mm_eligible(a, weight, a_scale, w_scale, bias) -> bool:
    # torch._int_mm: M > 16, K and N multiples of 8 (cuBLASLt IMMA).
    return a.shape[0] > 16 and a.shape[1] % 8 == 0 and weight.shape[0] % 8 == 0 and a.is_contiguous()


@ops.int8_linear.register("cublas", device=DeviceSpec("cuda", min_capability=(7, 5)), when=_int_mm_eligible)
def _int8_linear_cublas(a, weight, a_scale, w_scale, bias):
    out = torch._int_mm(a, weight.t()).float() * (a_scale * w_scale)
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)
