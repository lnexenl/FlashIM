"""Triton implicit-GEMM backend for ``ops.conv2d_at_points`` (evaluating an odd ``k x k`` conv only at ``K`` integer
pixel coordinates, e.g. DISK's 128-ch 5x5 descriptor head at ``K <= 4096`` keypoints).

Avoids materializing the ``[Cin, K, k, k]`` intermediate (32 MB FP32 at DISK's shape) by gathering ``[BLOCK_P, BLOCK_C]``
taps directly into full-FP32 (``tf32x3``) ``tl.dot`` accumulators. Kept in its own module to stay decoupled from the
dense quantized conv kernels in ``backends/triton/conv.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


@triton.jit
def _conv2d_at_points_kernel(
    x_ptr, w_ptr, bias_ptr, pts_ptr, out_ptr,
    K, CIN, COUT, H, W, sx_c, sx_h, sx_w, sp_0, sp_1,
    KH: tl.constexpr, KW: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """``w_ptr`` is packed ``[KH * KW, CIN, COUT]`` contiguous FP32 so every ``[BLOCK_C, BLOCK_N]`` load is coalesced."""
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    p_ok = offs_p < K
    n_ok = offs_n < COUT

    px = tl.load(pts_ptr + offs_p.to(tl.int64) * sp_0, mask=p_ok, other=0).to(tl.int32)
    py = tl.load(pts_ptr + offs_p.to(tl.int64) * sp_0 + sp_1, mask=p_ok, other=0).to(tl.int32)

    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
    offs_c_base = tl.arange(0, BLOCK_C)
    rad_h: tl.constexpr = KH // 2
    rad_w: tl.constexpr = KW // 2

    for ky in tl.static_range(KH):
        iy = py + (ky - rad_h)
        y_ok = p_ok & (iy >= 0) & (iy < H)
        for kx in tl.static_range(KW):
            ix = px + (kx - rad_w)
            pt_ok = y_ok & (ix >= 0) & (ix < W)
            tap = ky * KW + kx
            base_pix = iy.to(tl.int64) * sx_h + ix.to(tl.int64) * sx_w
            w_tap_base = w_ptr + (tap * CIN).to(tl.int64) * COUT
            for c0 in range(0, CIN, BLOCK_C):
                offs_c = c0 + offs_c_base
                c_ok = offs_c < CIN
                a = tl.load(
                    x_ptr + base_pix[:, None] + offs_c[None, :].to(tl.int64) * sx_c,
                    mask=pt_ok[:, None] & c_ok[None, :], other=0.0,
                ).to(tl.float32)
                wt = tl.load(
                    w_tap_base + offs_c[:, None].to(tl.int64) * COUT + offs_n[None, :],
                    mask=c_ok[:, None] & n_ok[None, :], other=0.0,
                )
                acc = tl.dot(a, wt, acc, input_precision="tf32x3")

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_ok, other=0.0)[None, :]
    tl.store(
        out_ptr + offs_p[:, None].to(tl.int64) * COUT + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=p_ok[:, None] & n_ok[None, :],
    )


def _conv2d_at_points_triton(x: Tensor, weight: Tensor, bias: Tensor | None, points: Tensor) -> Tensor:
    _, cin, h, w = x.shape
    cout, _, kh, kw = weight.shape
    k = points.shape[0]
    if k == 0:
        return weight.new_empty(0, cout)
    # [Cout, Cin, kh, kw] -> [kh*kw, Cin, Cout] FP32 contiguous (1 MB for DISK's 128x80x5x5 -> ~2 us)
    w_packed = weight.float().permute(2, 3, 1, 0).contiguous()
    b_fp32 = bias.float().contiguous() if bias is not None else w_packed
    out = torch.empty(k, cout, device=x.device, dtype=weight.dtype)

    block_p = 64
    block_n = min(64, max(16, triton.next_power_of_2(cout)))
    block_c = 32 if cin >= 32 else 16
    grid = (triton.cdiv(k, block_p), triton.cdiv(cout, block_n))
    _conv2d_at_points_kernel[grid](
        x, w_packed, b_fp32, points, out,
        k, cin, cout, h, w, x.stride(1), x.stride(2), x.stride(3), points.stride(0), points.stride(1),
        KH=kh, KW=kw, HAS_BIAS=bias is not None,
        BLOCK_P=block_p, BLOCK_N=block_n, BLOCK_C=block_c, num_warps=4,
    )
    return out


def _conv2d_at_points_eligible(x: Tensor, weight: Tensor, bias: Tensor | None, points: Tensor) -> bool:
    return (
        x.is_cuda and x.ndim == 4 and x.shape[0] == 1
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and weight.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and weight.shape[1] == x.shape[1]
        and weight.shape[2] % 2 == 1 and weight.shape[3] % 2 == 1
        and weight.shape[2] * weight.shape[3] <= 49
        and points.ndim == 2 and points.shape[1] == 2
    )


ops.conv2d_at_points.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_conv2d_at_points_eligible
)(_conv2d_at_points_triton)
