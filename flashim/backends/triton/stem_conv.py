"""Triton stem & pooled-epilogue conv kernels for classic CNN extractors (SuperPoint, ALIKE).

Kept in its own module separate from ``backends/triton/conv.py`` (dense conv autotune & chaining).

- ``conv3x3_small_cin_relu``: 3x3 / pad-1 BF16 stem conv for ``Cin in (1, 3)`` on BF16 tensor cores
  (``K=16`` for ``Cin=1``, ``K=32`` for ``Cin=3``) with fused bias + ReLU and BF16 NHWC, static INT8 NHWC,
  or static MXFP8 NHWC output. Replaces cuDNN's slow SIMT ``1->64`` stem in SuperPoint (1.73 -> ~0.18 ms).
- ``int8_conv2d_maxpool_only``: implicit-GEMM INT8 conv + ReLU + 2x2/stride-2 maxpool + static INT8 quantize
  without allocating or writing the unpooled full-resolution BF16 map (SuperPoint ``conv1b/2b/3b``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim import ops
from flashim.backends.triton.quantize import to_e4m3
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature, platform_for
from flashim.quant.tensor import Int8Tensor, QuantizedTensor

# -- small-Cin 3x3 + bias + ReLU (+ optional INT8 / MXFP8 quantize) ------------------------------------


@triton.jit
def _conv3x3_cin1_fma_kernel(
    x_ptr, w_ptr, bias_ptr, scale_ptr, out_ptr,
    H, W,
    COUT: tl.constexpr, SEG: tl.constexpr, OUT_MODE: tl.constexpr,
):
    """3x3 / pad-1 BF16 conv of a 1-channel NCHW tensor via exact 9-tap row-major FP32 FMA.
    DRAM-write-bound (~0.04 ms at 480x640, ~0.20 ms at 896x1200 on sm_120, 7-8.5x faster than cuDNN)
    while matching cuDNN's SIMT accumulation order bit-for-bit at 480x640 (0 / 19.6M diffs)."""
    row = tl.program_id(0)  # b * H + y
    b = row // H
    oh = row - b * H
    ow = tl.program_id(1) * SEG + tl.arange(0, SEG)
    ow_ok = ow < W
    n = tl.arange(0, COUT)

    acc = tl.zeros((SEG, COUT), dtype=tl.float32)
    base_b = b.to(tl.int64) * H * W
    for ky in tl.static_range(3):
        ih = oh + ky - 1
        ih_ok = (ih >= 0) & (ih < H)
        for kx in tl.static_range(3):
            iw = ow + kx - 1
            ok = ow_ok & ih_ok & (iw >= 0) & (iw < W)
            a = tl.load(x_ptr + base_b + ih.to(tl.int64) * W + iw, mask=ok, other=0.0).to(tl.float32)
            tap = ky * 3 + kx
            wt = tl.load(w_ptr + tap * COUT + n).to(tl.float32)
            acc = tl.fma(a[:, None], wt[None, :], acc)

    value = (acc.to(tl.bfloat16).to(tl.float32) + tl.load(bias_ptr + n).to(tl.float32)[None, :]).to(tl.bfloat16)
    value = tl.maximum(value, 0.0)

    out_ptrs = out_ptr + (row.to(tl.int64) * W + ow[:, None].to(tl.int64)) * COUT + n[None, :]
    if OUT_MODE == 1:
        scaled = libdevice.rint(value.to(tl.float32) * (1.0 / tl.load(scale_ptr)))
        q = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
        tl.store(out_ptrs, q, mask=ow_ok[:, None])
    elif OUT_MODE == 2:
        exp = tl.load(scale_ptr + n // 32).to(tl.float32)
        scaled = value.to(tl.float32) * tl.exp2(-exp)[None, :]
        q = to_e4m3(tl.maximum(tl.minimum(scaled, 448.0), -448.0))
        tl.store(out_ptrs, q, mask=ow_ok[:, None])
    else:
        tl.store(out_ptrs, value, mask=ow_ok[:, None])


@triton.jit
def _conv3x3_small_cin_kernel(
    x_ptr, w_ptr, bias_ptr, scale_ptr, out_ptr,
    H, W,
    CIN: tl.constexpr, K_REAL: tl.constexpr, K_PAD: tl.constexpr,
    COUT: tl.constexpr, SEG: tl.constexpr, OUT_MODE: tl.constexpr,
):
    """3x3 / pad-1 BF16 conv of a ``CIN in (1, 3)`` NCHW tensor as one ``tl.dot`` per ``SEG``-pixel row tile.
    ``OUT_MODE``: ``0`` = BF16 NHWC, ``1`` = static INT8 NHWC, ``2`` = static MXFP8 E4M3 NHWC."""
    row = tl.program_id(0)  # b * H + y
    b = row // H
    oh = row - b * H

    k = tl.arange(0, K_PAD)
    if CIN == 1:
        ci = tl.zeros((K_PAD,), dtype=tl.int32)
        r = k
    else:
        ci = k // 9
        r = k % 9
    ky = r // 3 - 1
    kx = r % 3 - 1

    ow = tl.program_id(1) * SEG + tl.arange(0, SEG)
    ih = oh + ky[None, :]
    iw = ow[:, None] + kx[None, :]
    ow_ok = ow < W
    ok = ow_ok[:, None] & (k < K_REAL)[None, :] & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    a = tl.load(x_ptr + ((b * CIN + ci[None, :]).to(tl.int64) * H + ih) * W + iw, mask=ok, other=0.0)

    n = tl.arange(0, COUT)
    wt = tl.load(w_ptr + k[:, None] * COUT + n[None, :])
    acc = tl.dot(a, wt)
    value = (acc.to(tl.bfloat16).to(tl.float32) + tl.load(bias_ptr + n).to(tl.float32)[None, :]).to(tl.bfloat16)
    value = tl.maximum(value, 0.0)

    out_ptrs = out_ptr + (row.to(tl.int64) * W + ow[:, None].to(tl.int64)) * COUT + n[None, :]
    if OUT_MODE == 1:
        scaled = libdevice.rint(value.to(tl.float32) * (1.0 / tl.load(scale_ptr)))
        q = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
        tl.store(out_ptrs, q, mask=ow_ok[:, None])
    elif OUT_MODE == 2:
        exp = tl.load(scale_ptr + n // 32).to(tl.float32)
        scaled = value.to(tl.float32) * tl.exp2(-exp)[None, :]
        q = to_e4m3(tl.maximum(tl.minimum(scaled, 448.0), -448.0))
        tl.store(out_ptrs, q, mask=ow_ok[:, None])
    else:
        tl.store(out_ptrs, value, mask=ow_ok[:, None])


def _conv3x3_small_cin_prepare(weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """Pack ``[COUT, CIN, 3, 3]`` -> ``[K_PAD, COUT]`` contiguous BF16 once per :class:`BoundOp`."""
    cout, cin = weight.shape[0], weight.shape[1]
    k_real = cin * 9
    k_pad = 16 if cin == 1 else 32
    packed = F.pad(weight.detach().reshape(cout, k_real), (0, k_pad - k_real)).t().contiguous()
    return packed, bias.detach().contiguous()


def _conv3x3_small_cin_accepts(weight: Tensor, bias: Tensor) -> bool:
    if weight.ndim != 4 or weight.shape[1] not in (1, 3) or weight.shape[2:] != (3, 3):
        return False
    cout = weight.shape[0]
    return (
        weight.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16
        and 16 <= cout <= 128 and (cout & (cout - 1)) == 0 and bias.shape == (cout,)
    )


def _conv3x3_small_cin_triton(
    x: Tensor,
    packed: Tensor,
    bias_c: Tensor,
    *,
    out_scale: Tensor | None = None,
    out_exponent: Tensor | None = None,
) -> Tensor | Int8Tensor | QuantizedTensor:
    b, cin, h, w = x.shape
    cout = packed.shape[1]
    k_real = cin * 9
    k_pad = 16 if cin == 1 else 32

    if out_scale is not None:
        out_mode = 1
        out = torch.empty(b, h, w, cout, device=x.device, dtype=torch.int8)
        scale_ptr = out_scale
    elif out_exponent is not None:
        out_mode = 2
        out = torch.empty(b, h, w, cout, device=x.device, dtype=torch.float8_e4m3fn)
        scale_ptr = out_exponent
    else:
        out_mode = 0
        out = torch.empty(b, h, w, cout, device=x.device, dtype=torch.bfloat16)
        scale_ptr = bias_c

    if cin == 1:
        seg = 64
        _conv3x3_cin1_fma_kernel[(b * h, triton.cdiv(w, seg))](
            x, packed, bias_c, scale_ptr, out, h, w,
            COUT=cout, SEG=seg, OUT_MODE=out_mode, num_warps=4,
        )
    else:
        seg = 128
        _conv3x3_small_cin_kernel[(b * h, triton.cdiv(w, seg))](
            x, packed, bias_c, scale_ptr, out, h, w,
            CIN=cin, K_REAL=k_real, K_PAD=k_pad, COUT=cout, SEG=seg, OUT_MODE=out_mode, num_warps=4,
        )
    if out_mode == 1:
        return Int8Tensor(out, out_scale)
    if out_mode == 2:
        return QuantizedTensor(out, out_exponent, (b, h, w, cout))
    return out.permute(0, 3, 1, 2)


def _conv3x3_small_cin_eligible(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    out_scale: Tensor | None = None,
    out_exponent: Tensor | None = None,
) -> bool:
    if not (
        x.is_cuda and x.ndim == 4 and x.shape[1] == weight.shape[1]
        and x.dtype == torch.bfloat16 and x.is_contiguous()
    ):
        return False
    return out_exponent is None or (
        weight.shape[0] % 32 == 0 and platform_for(x.device).supports(Feature.FP8_E4M3)
    )


ops.conv3x3_small_cin_relu.register(
    "triton", device=DeviceSpec("cuda", min_capability=(8, 0)), requires=[Feature.TRITON],
    when=_conv3x3_small_cin_eligible, accepts=_conv3x3_small_cin_accepts, prepare=_conv3x3_small_cin_prepare,
)(_conv3x3_small_cin_triton)


# -- INT8 implicit-GEMM conv + ReLU + MaxPool2d(2) -> INT8 (pool-only, no unpooled BF16 store) ----------


@triton.jit
def _int8_conv2d_maxpool_only_kernel(
    x_ptr, w_ptr, bias_ptr, pool_ptr, out_scale_ptr, a_scale_ptr, w_scale_ptr,
    B, H, W, HO, WO,
    CIN: tl.constexpr, COUT: tl.constexpr, TAPS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    HAS_BIAS: tl.constexpr, RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Same 2-row x ``BLOCK_M // 2`` pooling-window tile as ``_int8_conv2d_kernel(POOL=True)``, omitting the
    full-resolution BF16 store: bit-identical pooled INT8 output with 4x fewer output writes."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(COUT, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    SEG: tl.constexpr = BLOCK_M // 2
    segs = tl.cdiv(WO, SEG)
    pb = pid_m // ((HO // 2) * segs)
    b = pb + tl.zeros((BLOCK_M,), dtype=tl.int32)
    py = (pid_m // segs) % (HO // 2)
    seg = pid_m % segs
    m = tl.arange(0, BLOCK_M)
    oh = 2 * py + m // SEG
    ow = seg * SEG + m % SEG
    mask_m = (ow < WO) & (b < B)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < COUT
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for ky in tl.static_range(KH):
        ih = oh * SH + ky * DH - PH
        ih_ok = (ih >= 0) & (ih < H)
        for kx in tl.static_range(KW):
            iw = ow * SW + kx * DW - PW
            row_ok = mask_m & ih_ok & (iw >= 0) & (iw < W)
            pixel = ((b * H + ih) * W + iw).to(tl.int64) * CIN
            tap = ky * KW + kx
            for k0 in tl.static_range(0, CIN, BLOCK_K):
                offs_c = k0 + offs_k
                c_ok = offs_c < CIN
                a = tl.load(x_ptr + pixel[:, None] + offs_c[None, :], mask=row_ok[:, None] & c_ok[None, :], other=0)
                b_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * (TAPS * CIN) + tap * CIN + offs_c[:, None]
                wt = tl.load(b_ptrs, mask=mask_n[None, :] & c_ok[:, None], other=0)
                accumulator = tl.dot(a, wt, accumulator, out_dtype=tl.int32)

    scale = tl.load(a_scale_ptr) * tl.load(w_scale_ptr + offs_n, mask=mask_n, other=0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = tl.fma(accumulator.to(tl.float32), scale[None, :], bias[None, :])
    else:
        out = accumulator.to(tl.float32) * scale[None, :]
    if RELU:
        out = tl.maximum(out, 0.0)

    pooled = tl.max(tl.reshape(out.to(tl.bfloat16).to(tl.float32), (2, SEG // 2, 2, BLOCK_N)), axis=2)
    pooled = tl.max(pooled, axis=0)
    qx = seg * (SEG // 2) + tl.arange(0, SEG // 2)
    scaled = libdevice.rint(pooled * (1.0 / tl.load(out_scale_ptr)))
    pooled_ptrs = pool_ptr + ((pb * (HO // 2) + py) * (WO // 2) + qx).to(tl.int64)[:, None] * COUT + offs_n[None, :]
    tl.store(
        pooled_ptrs,
        tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8),
        mask=(qx < WO // 2)[:, None] & mask_n[None, :],
    )


def _int8_conv2d_maxpool_only_triton(
    x, weight, a_scale, w_scale, bias, kh, kw, sh, sw, ph, pw, dh, dw, *, relu=False, out_scale
) -> Tensor:
    b, h, w, cin = x.shape
    cout = weight.shape[0]
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    pooled = torch.empty(b, ho // 2, wo // 2, cout, device=x.device, dtype=torch.int8)
    # Fixed tile (128x64 or 64x64): avoids Triton Autotuner CPU dispatch overhead on every eager call.
    block_m = 128 if ho * wo >= 65536 else 64
    block_n = 64 if cout >= 64 else 32
    block_k = 64 if cin >= 64 else max(32, triton.next_power_of_2(cin))
    grid = (b * (ho // 2) * triton.cdiv(wo, block_m // 2) * triton.cdiv(cout, block_n),)
    _int8_conv2d_maxpool_only_kernel[grid](
        x, weight, bias if bias is not None else a_scale, pooled, out_scale, a_scale, w_scale,
        b, h, w, ho, wo,
        CIN=cin, COUT=cout, TAPS=kh * kw, KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw, DH=dh, DW=dw,
        HAS_BIAS=bias is not None, RELU=relu,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4, num_stages=3,
    )
    return pooled


def _int8_conv2d_maxpool_only_eligible(x, weight, a_scale, w_scale, bias, kh, kw, sh, sw, ph, pw, dh, dw, **_) -> bool:
    ho = (x.shape[1] + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (x.shape[2] + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return (
        x.is_cuda and x.ndim == 4 and x.dtype == torch.int8 and x.is_contiguous()
        and weight.shape == (weight.shape[0], kh * kw * x.shape[-1])
        and kh * kw <= 49 and ho % 2 == 0 and wo % 2 == 0
    )


ops.int8_conv2d_maxpool_only.register(
    "triton", device=DeviceSpec("cuda", min_capability=(7, 5)), requires=[Feature.TRITON],
    when=_int8_conv2d_maxpool_only_eligible,
)(_int8_conv2d_maxpool_only_triton)
