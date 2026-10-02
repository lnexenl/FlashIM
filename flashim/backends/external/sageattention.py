"""SageAttention (thu-ml; Q/K INT8 with K smoothing) as ``sdpa`` impls; loaded only when the package is installed.

Opt-in (priority -1): pin with ``ops.sdpa.get("sage")`` (INT8 QK + FP16 PV, FP32 accumulation) or ``"sage-fp8"``
(INT8 QK + FP8 PV). Head dims 64 to 128 are accepted: SageAttention's CUDA entry points zero-pad dims between
64 and 128 to 128 themselves (e.g. 96 for XFeat's LighterGlue) and default the softmax scale to the original dim's.
"""

from __future__ import annotations

import sageattention
import torch
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec

_HEAD_DIMS = (64, 96, 128)


def _eligible(q: Tensor, k: Tensor, v: Tensor) -> bool:
    return (
        q.ndim == 4
        and q.shape == k.shape == v.shape
        and q.shape[-1] in _HEAD_DIMS
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    )


@torch.library.custom_op("flashim::sage_sdpa", mutates_args=())
def _sage_sdpa(q: Tensor, k: Tensor, v: Tensor, variant: str) -> Tensor:
    if variant == "fp8":
        return sageattention.sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, tensor_layout="HND", is_causal=False, pv_accum_dtype="fp32+fp32"
        )
    return sageattention.sageattn_qk_int8_pv_fp16_cuda(
        q, k, v, tensor_layout="HND", is_causal=False, pv_accum_dtype="fp32"
    )


@_sage_sdpa.register_fake
def _(q: Tensor, k: Tensor, v: Tensor, variant: str) -> Tensor:
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


@ops.sdpa.register("sage", device=DeviceSpec("cuda", min_capability=(8, 0)), priority=-1, when=_eligible)
def _sdpa_sage(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return _sage_sdpa(q, k, v, "fp16")


@ops.sdpa.register("sage-fp8", device=DeviceSpec("cuda", min_capability=(8, 9)), priority=-1, when=_eligible)
def _sdpa_sage_fp8(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return _sage_sdpa(q, k, v, "fp8")
