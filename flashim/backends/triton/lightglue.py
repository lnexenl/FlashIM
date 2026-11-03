"""Triton impls of the LightGlue glue ops (:mod:`flashim.ops.lightglue`): the INT8 GEMM of
:func:`~flashim.backends.triton.gemm.int8_gemm_store` with LightGlue's store layouts. Bit-identical to the unfused
modules (``int8_linear`` + view / permute + ``rope_qk_interleaved``; ``int8_linear`` + ``flip``), as measured on an L20
(sm_89) and an H20 (sm_90) on LoMa's matcher inputs."""

from __future__ import annotations

import torch
from torch import Tensor

from flashim import ops
from flashim.backends.triton.gemm import (
    STORE_BOTH,
    STORE_QKV_ROPE,
    STORE_SWAPPED,
    _static_gemm_eligible,
    int8_gemm_store,
)
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


def _qkv_rope_eligible(a, weight, a_scale, w_scale, bias, cos, sin, heads) -> bool:
    if not _static_gemm_eligible(a, weight, a_scale, w_scale, bias) or cos.ndim != 4:
        return False
    batch, _, tokens, dim = cos.shape
    return (
        cos.shape == sin.shape and cos.shape[1] == 1 and cos.is_contiguous() and sin.is_contiguous()
        and cos.dtype == sin.dtype and a.shape[0] == batch * tokens and weight.shape[0] == 3 * heads * dim
        and dim % 2 == 0
    )


@ops.lightglue_qkv_rope.register(
    "triton", device=DeviceSpec("cuda", min_capability=(7, 5)), requires=[Feature.TRITON], when=_qkv_rope_eligible
)
def _lightglue_qkv_rope_impl(a, weight, a_scale, w_scale, bias, cos, sin, heads) -> Tensor:
    batch, _, tokens, dim = cos.shape
    out = torch.empty(3, batch, heads, tokens, dim, device=a.device, dtype=torch.bfloat16)
    int8_gemm_store(a, weight, a_scale, w_scale, bias, STORE_QKV_ROPE, out=out, aux=cos, aux2=sin, half=tokens,
                    heads=heads, head_dim=dim)
    return out


@ops.lightglue_swapped_linear.register(
    "triton", device=DeviceSpec("cuda", min_capability=(7, 5)), requires=[Feature.TRITON],
    when=lambda a, weight, a_scale, w_scale, bias, *, keep=True: (
        _static_gemm_eligible(a, weight, a_scale, w_scale, bias) and a.shape[0] % 2 == 0
    ),
)
def _lightglue_swapped_linear_impl(a, weight, a_scale, w_scale, bias, *, keep: bool = True):
    rows, n = a.shape[0], weight.shape[0]
    swapped = torch.empty(rows, n, device=a.device, dtype=torch.bfloat16)
    out = torch.empty_like(swapped) if keep else swapped
    int8_gemm_store(a, weight, a_scale, w_scale, bias, STORE_BOTH if keep else STORE_SWAPPED, out=out, out2=swapped,
                    half=rows // 2)
    return (out if keep else None), swapped
