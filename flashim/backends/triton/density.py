"""Tiled Gaussian KDE: each program owns ``BLOCK_M`` query points and streams all points."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


@triton.jit
def _kde_kernel(x_ptr, out_ptr, n, neg_scale_log2, D: tl.constexpr, DP: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, DP)
    row_ok = rows < n
    xi = tl.load(x_ptr + rows[:, None] * D + dims[None, :], mask=row_ok[:, None] & (dims[None, :] < D), other=0.0)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in range(0, n, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_ok = cols < n
        d2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for c in tl.static_range(D):
            xc = tl.sum(tl.where(dims[None, :] == c, xi, 0.0), axis=1)          # [BLOCK_M]
            yc = tl.load(x_ptr + cols * D + c, mask=col_ok, other=0.0)          # [BLOCK_N]
            diff = xc[:, None] - yc[None, :]
            d2 += diff * diff
        acc += tl.sum(tl.where(col_ok[None, :], tl.exp2(d2 * neg_scale_log2), 0.0), axis=1)
    tl.store(out_ptr + rows, acc, mask=row_ok)


def kde_density(x: Tensor, std: float = 0.1, *, block_m: int = 64, block_n: int = 128, num_warps: int = 4) -> Tensor:
    n, d = x.shape
    x = x.float().contiguous()
    out = torch.empty(n, device=x.device, dtype=torch.float32)
    neg_scale_log2 = -1.0 / (2 * std * std) * math.log2(math.e)
    _kde_kernel[(triton.cdiv(n, block_m),)](
        x, out, n, neg_scale_log2, D=d, DP=triton.next_power_of_2(d), BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out

# -- registrations ---------------------------------------------------------------------------------------

def _eligible(x: Tensor, std: float = 0.1) -> bool:
    return x.ndim == 2 and 1 <= x.shape[1] <= 8 and x.shape[0] > 0


@ops.kde_density.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _kde_density_impl(x: Tensor, std: float = 0.1) -> Tensor:
    """Tiled FP32 evaluation with no ``N x N`` intermediate; returns FP16 like the reference."""
    return kde_density(x, std).half()
