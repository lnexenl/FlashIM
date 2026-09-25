"""RoMaV2 refiner feature projection: BF16 features -> FP32 ``Linear``.

The reference converts to FP32 and runs an FP32 GEMM (full precision; the projected
features feed FP32 sampling and correlation). The Triton path covers the small-output
projections (``P <= 16``, the stride-1 refiner), where the FP32 GEMM is inefficient;
larger projections stay on cuBLAS, which is as fast as a hand-written kernel there.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op


def _refiner_proj_reference(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return F.linear(x.float(), weight, bias)


refiner_proj = Op("refiner_proj", _refiner_proj_reference)


def _eligible(x: Tensor, weight: Tensor, bias: Tensor) -> bool:
    k = x.shape[-1]
    return (
        x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.shape[0] <= 16
        and weight.shape[1] == k
        and k in (16, 32, 64, 128)
        and x.numel() < 2**31
    )


@refiner_proj.register("triton", device=DeviceSpec("cuda"), when=_eligible)
def _refiner_proj_triton(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    from flashim.ops._triton.refiner import refiner_proj as kernel

    return kernel(x, weight, bias)
