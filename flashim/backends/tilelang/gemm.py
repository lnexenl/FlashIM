"""Block-scaled MXFP8 GEMM (``ops.mxfp8_linear``) for the large-``K`` shapes cuBLAS's ``F.scaled_mm`` rejects
(padded ``N`` or ``K`` not a multiple of 128) and where it measurably beats the Triton fallback — DISK's and
LoFTR's widest convolutions (K up to 3200); see ``_gemm_kernels.py`` (loaded on first call) for the kernel, the
threshold, and why smaller-K layers stay on Triton."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature, platform_for
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight

BLOCK_M = 64
MIN_K = 2000  # below this, measured roughly break-even or slightly behind Triton (see _gemm_kernels.py)


def _kernels():
    from flashim.backends.tilelang import _gemm_kernels

    return _gemm_kernels


def _prepare(weight: QuantizedWeight) -> tuple[Tensor, Tensor]:
    return weight.values.contiguous(), torch.exp2(weight.scale_exponent.float())


def _cublas_eligible(weight: QuantizedWeight) -> bool:
    return weight.out_features % 128 == 0 and weight.in_features % 128 == 0


def _accepts(weight: QuantizedWeight) -> bool:
    """Not on Hopper, where Triton's GEMM is at least as fast with the same bits (H20, K = 3200 / 2016 / 2304:
    0.222 / 0.324 / 0.403 ms against 0.226 / 0.423 / 0.464 ms here at 67200 rows, 0.818 / 1.254 / 1.581 against 0.837 /
    1.628 / 1.796 ms at 268800 rows, and 2-3x at 4000 rows)."""
    if platform_for(weight.values.device).capability == (9, 0):
        return False
    return weight.in_features >= MIN_K and not _cublas_eligible(weight)


@torch.library.custom_op("flashim::mxfp8_gemm_tilelang", mutates_args=())
def _mxfp8_gemm(a: Tensor, a_scale: Tensor, weight_values: Tensor, weight_scale: Tensor) -> Tensor:
    """Split into a zero-copy "bulk" launch (the largest multiple of ``BLOCK_M`` rows, read directly from ``a`` —
    no padding) plus, if ``m`` isn't itself a multiple, one small launch for the ``< BLOCK_M``-row remainder.

    A single ``F.pad`` up to the next multiple of ``BLOCK_M`` looked free (at most 63 extra rows) but isn't:
    padding a tensor allocates a new one and copies *all* of it, so it was a full ``O(M*K)`` memory pass — for
    LoFTR's ~130k-row, K=2016 activations, as expensive as the GEMM itself. This keeps the pad (if any) to the
    remainder alone, ``O(BLOCK_M * K)`` — a fixed, tiny cost regardless of ``M``.
    """
    m, k = a.shape
    n = weight_values.shape[0]
    bulk_m = (m // BLOCK_M) * BLOCK_M
    out = torch.empty(m, n, device=a.device, dtype=torch.float32)
    if bulk_m:
        _kernels().mxfp8_gemm_kernel(bulk_m, n, k)(a[:bulk_m], weight_values, a_scale, weight_scale, out[:bulk_m])
    if remainder := m - bulk_m:
        tail = F.pad(a[bulk_m:].view(torch.uint8), (0, 0, 0, BLOCK_M - remainder)).view(a.dtype)
        tail_out = torch.empty(BLOCK_M, n, device=a.device, dtype=torch.float32)
        _kernels().mxfp8_gemm_kernel(BLOCK_M, n, k)(tail, weight_values, a_scale, weight_scale, tail_out)
        out[bulk_m:] = tail_out[:remainder]
    return out


@_mxfp8_gemm.register_fake
def _(a: Tensor, a_scale: Tensor, weight_values: Tensor, weight_scale: Tensor) -> Tensor:
    return a.new_empty((a.shape[0], weight_values.shape[0]), dtype=torch.float32)


def _mxfp8_linear_tilelang(
    activation: QuantizedTensor, bias: Tensor | None, weight_values: Tensor, weight_scale: Tensor
) -> Tensor:
    a_scale = torch.exp2(activation.scale_exponent.float())
    out = _mxfp8_gemm(activation.matrix(), a_scale, weight_values, weight_scale)
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


ops.mxfp8_linear.register(
    "tilelang", device=DeviceSpec("cuda"), requires=[Feature.TILELANG, Feature.MMA_SYNC, Feature.FP8_E4M3],
    priority=1, prepare=_prepare, accepts=_accepts,
)(_mxfp8_linear_tilelang)
