"""Triton kernels for EfficientLoFTR fused ops:
- ``eloftr_upsample_gather_windows``: 2x bilinear upsample + window gather directly from 1/2-res FPN features
  (avoids materializing the 1/1 full-res feature map).
- ``eloftr_upsample_add_leaky_relu``: fused bilinear upsample + add + LeakyReLU (+ optional static INT8 / MXFP8
  quantization for coarse ``mlp.2`` and FPN ``outconv2`` inputs).
- ``eloftr_post_norm_add``: fused post-LayerNorm + residual add on NHWC token grids.
- ``eloftr_agg_norm`` / ``eloftr_pool_norm``: fused KxK depthwise conv or MaxPool2d + LayerNorm (+ optional static
  INT8, dynamic per-row INT8, or static MXFP8 quantization).
- ``eloftr_leaky_relu_quantize``: fused LeakyReLU + static INT8 / MXFP8 quantization at FPN conv boundaries.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from flashim import ops
from flashim.backends.triton.quantize import to_e4m3
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature, platform_for
from flashim.quant.formats.mx import padded_channels
from flashim.quant.tensor import Int8Tensor, QuantizedTensor

# -- 1) 2x upsample + fine window gather directly from 1/2-res feature map ---------------------------------


@triton.jit
def _upsample_gather_windows_kernel(
    feat_ptr,
    ids_ptr,
    out_ptr,
    sc,
    sh,
    sw,
    H_HALF,
    W_HALF,
    H_FULL,
    W_FULL,
    WC,
    C: tl.constexpr,
    WIN: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One program per ``(match m, window row dy)``: writes ``[WIN, C]`` directly into ``out[m, dy*WIN:(dy+1)*WIN]``."""
    m = tl.program_id(0).to(tl.int64)
    dy = tl.program_id(1)
    cell = tl.load(ids_ptr + m).to(tl.int32)
    cy = cell // WC
    cx = cell % WC

    fy = cy * 8 + dy - PAD
    fy_ok = (fy >= 0) & (fy < H_FULL)
    sy = tl.maximum(0.5 * (fy.to(tl.float32) + 0.5) - 0.5, 0.0)
    y0 = sy.to(tl.int32)
    y1 = tl.where(y0 < H_HALF - 1, y0 + 1, y0)
    ly1 = sy - y0.to(tl.float32)
    ly0 = 1.0 - ly1

    dx = tl.arange(0, BLOCK_W)
    dx_ok = dx < WIN
    fx = cx * 8 + dx - PAD
    fx_ok = dx_ok & (fx >= 0) & (fx < W_FULL) & fy_ok
    sx = tl.maximum(0.5 * (fx.to(tl.float32) + 0.5) - 0.5, 0.0)
    x0 = sx.to(tl.int32)
    x1 = tl.where(x0 < W_HALF - 1, x0 + 1, x0)
    lx1 = sx - x0.to(tl.float32)
    lx0 = 1.0 - lx1

    c = tl.arange(0, BLOCK_C)
    c_ok = c < C
    mask = fx_ok[:, None] & c_ok[None, :]

    row0 = y0.to(tl.int64) * sh
    row1 = y1.to(tl.int64) * sh
    col0 = x0.to(tl.int64) * sw
    col1 = x1.to(tl.int64) * sw
    ch = c.to(tl.int64) * sc

    v00 = tl.load(feat_ptr + row0 + col0[:, None] + ch[None, :], mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(feat_ptr + row0 + col1[:, None] + ch[None, :], mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(feat_ptr + row1 + col0[:, None] + ch[None, :], mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(feat_ptr + row1 + col1[:, None] + ch[None, :], mask=mask, other=0.0).to(tl.float32)

    val = ly0 * (lx0[:, None] * v00 + lx1[:, None] * v01) + ly1 * (lx0[:, None] * v10 + lx1[:, None] * v11)
    val = tl.where(fx_ok[:, None], val, 0.0)

    out_base = ((m * (WIN * WIN) + dy.to(tl.int64) * WIN) + dx.to(tl.int64))[:, None] * C + c[None, :]
    tl.store(out_ptr + out_base, val.to(out_ptr.dtype.element_ty), mask=dx_ok[:, None] & c_ok[None, :])


def _upsample_gather_windows_triton(
    feat_half: Tensor,
    ids: Tensor,
    hc: int,
    wc: int,
    window_size: int = 8,
    pad: int = 0,
) -> Tensor:
    _, c, h_half, w_half = feat_half.shape
    m = ids.shape[0]
    out = torch.empty((m, window_size * window_size, c), device=feat_half.device, dtype=feat_half.dtype)
    if m == 0:
        return out
    ids_c = ids.contiguous()
    block_w = triton.next_power_of_2(window_size)
    block_c = triton.next_power_of_2(c)
    _upsample_gather_windows_kernel[(m, window_size)](
        feat_half,
        ids_c,
        out,
        feat_half.stride(1),
        feat_half.stride(2),
        feat_half.stride(3),
        h_half,
        w_half,
        h_half * 2,
        w_half * 2,
        wc,
        C=c,
        WIN=window_size,
        PAD=pad,
        BLOCK_W=block_w,
        BLOCK_C=block_c,
        num_warps=2 if block_w * block_c <= 1024 else 4,
    )
    return out


def _upsample_gather_eligible(
    feat_half: Tensor, ids: Tensor, hc: int, wc: int, window_size: int = 8, pad: int = 0
) -> bool:
    return (
        feat_half.ndim == 4
        and feat_half.shape[0] == 1
        and feat_half.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and feat_half.shape[1] <= 256
        and window_size <= 16
        and feat_half.shape[2] == hc * 4
        and feat_half.shape[3] == wc * 4
    )


ops.eloftr_upsample_gather_windows.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_upsample_gather_eligible
)(_upsample_gather_windows_triton)


# -- 2) Fused bilinear upsample(m_proj) + x_proj + LeakyReLU (+ optional INT8 / MXFP8 quantize) -----------


@triton.jit
def _upsample_add_leaky_relu_kernel(
    x_ptr,
    m_ptr,
    out_ptr,
    scale_ptr,
    H,
    W,
    h,
    w,
    scale_h,
    scale_w,
    neg_slope,
    C: tl.constexpr,
    PADDED_C: tl.constexpr,
    QUANT_MODE: tl.constexpr,  # 0: dense, 1: static int8, 2: static mxfp8
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One ``[BLOCK_P pixels, BLOCK_C channels]`` tile of NHWC ``[B, H, W, C]``."""
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    b = tl.program_id(2).to(tl.int64)
    p_ok = p < H * W
    c_ok = c < C
    out_c_ok = c < PADDED_C
    ok = p_ok[:, None] & c_ok[None, :]

    y = p // W
    x = p % W

    sy = tl.maximum(scale_h * (y.to(tl.float32) + 0.5) - 0.5, 0.0)
    y0 = sy.to(tl.int32)
    y1 = tl.where(y0 < h - 1, y0 + 1, y0)
    ly1 = sy - y0.to(tl.float32)
    ly0 = 1.0 - ly1

    sx = tl.maximum(scale_w * (x.to(tl.float32) + 0.5) - 0.5, 0.0)
    x0 = sx.to(tl.int32)
    x1 = tl.where(x0 < w - 1, x0 + 1, x0)
    lx1 = sx - x0.to(tl.float32)
    lx0 = 1.0 - lx1

    m_base = m_ptr + b * (h * w * C)
    v00 = tl.load(m_base + (y0 * w + x0)[:, None].to(tl.int64) * C + c[None, :], mask=ok, other=0.0).to(tl.float32)
    v01 = tl.load(m_base + (y0 * w + x1)[:, None].to(tl.int64) * C + c[None, :], mask=ok, other=0.0).to(tl.float32)
    v10 = tl.load(m_base + (y1 * w + x0)[:, None].to(tl.int64) * C + c[None, :], mask=ok, other=0.0).to(tl.float32)
    v11 = tl.load(m_base + (y1 * w + x1)[:, None].to(tl.int64) * C + c[None, :], mask=ok, other=0.0).to(tl.float32)
    up = ly0[:, None] * (lx0[:, None] * v00 + lx1[:, None] * v01) + ly1[:, None] * (
        lx0[:, None] * v10 + lx1[:, None] * v11
    )

    x_off = (b * (H * W) + p.to(tl.int64))[:, None] * C + c[None, :]
    xv = tl.load(x_ptr + x_off, mask=ok, other=0.0).to(tl.float32)
    val = xv + up
    val = tl.where(val >= 0.0, val, val * neg_slope)

    out_off = (b * (H * W) + p.to(tl.int64))[:, None] * PADDED_C + c[None, :]
    store_mask = p_ok[:, None] & out_c_ok[None, :]
    if QUANT_MODE == 0:
        tl.store(out_ptr + out_off, val.to(out_ptr.dtype.element_ty), mask=store_mask)
    elif QUANT_MODE == 1:
        dense = val.to(tl.bfloat16).to(tl.float32)
        inv = 1.0 / tl.load(scale_ptr)
        scaled = libdevice.rint(dense * inv)
        q = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
        tl.store(out_ptr + out_off, q, mask=store_mask)
    else:
        dense = tl.where(c_ok[None, :], val.to(tl.bfloat16).to(tl.float32), 0.0)
        exp = tl.load(scale_ptr + c // 32, mask=out_c_ok, other=0).to(tl.float32)
        scaled = dense * tl.exp2(-exp)[None, :]
        q = to_e4m3(tl.maximum(tl.minimum(scaled, 448.0), -448.0))
        tl.store(out_ptr + out_off, q, mask=store_mask)


def _upsample_add_leaky_relu_triton(
    x_proj: Tensor,
    m_proj: Tensor,
    negative_slope: float = 0.01,
    *,
    int8_scale: Tensor | None = None,
    mxfp8_exponent: Tensor | None = None,
) -> Tensor | Int8Tensor | QuantizedTensor:
    b, H, W, c = x_proj.shape
    _, h, w, _ = m_proj.shape
    x_c = x_proj.contiguous()
    m_c = m_proj.contiguous()
    if int8_scale is not None:
        quant_mode = 1
        padded_c = c
        out = torch.empty((b, H, W, c), device=x_c.device, dtype=torch.int8)
        scale_t = int8_scale
    elif mxfp8_exponent is not None:
        quant_mode = 2
        padded_c = padded_channels(c)
        out = torch.empty((b, H, W, padded_c), device=x_c.device, dtype=torch.float8_e4m3fn)
        scale_t = mxfp8_exponent
    else:
        quant_mode = 0
        padded_c = c
        out = torch.empty_like(x_c)
        scale_t = x_c

    block_p, block_c = 32, min(128, triton.next_power_of_2(padded_c))
    grid = (triton.cdiv(H * W, block_p), triton.cdiv(padded_c, block_c), b)
    _upsample_add_leaky_relu_kernel[grid](
        x_c,
        m_c,
        out,
        scale_t,
        H,
        W,
        h,
        w,
        h / H,
        w / W,
        float(negative_slope),
        C=c,
        PADDED_C=padded_c,
        QUANT_MODE=quant_mode,
        BLOCK_P=block_p,
        BLOCK_C=block_c,
        num_warps=4,
    )
    if quant_mode == 1:
        return Int8Tensor(out, int8_scale)
    if quant_mode == 2:
        return QuantizedTensor(out, mxfp8_exponent, (b, H, W, c))
    return out


def _upsample_add_leaky_relu_eligible(
    x_proj: Tensor,
    m_proj: Tensor,
    negative_slope: float = 0.01,
    *,
    int8_scale: Tensor | None = None,
    mxfp8_exponent: Tensor | None = None,
) -> bool:
    if x_proj.ndim != 4 or m_proj.ndim != 4 or x_proj.shape[0] != m_proj.shape[0] or x_proj.shape[3] != m_proj.shape[3]:
        return False
    if x_proj.dtype not in (torch.float32, torch.bfloat16, torch.float16) or m_proj.dtype != x_proj.dtype:
        return False
    if int8_scale is not None and mxfp8_exponent is not None:
        return False
    if mxfp8_exponent is not None and not platform_for(x_proj.device).supports(Feature.FP8_E4M3):
        return False
    return not ((mxfp8_exponent is not None or int8_scale is not None) and x_proj.dtype != torch.bfloat16)


ops.eloftr_upsample_add_leaky_relu.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_upsample_add_leaky_relu_eligible
)(_upsample_add_leaky_relu_triton)


# -- 3) Fused post-LayerNorm + residual add: x + LayerNorm(m) ---------------------------------------------


@triton.jit
def _post_norm_add_kernel(
    m_ptr,
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    eps,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One program per token row: ``out = x + LayerNorm(m, w, b, eps)``."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C
    off = row * C + cols

    mv = tl.load(m_ptr + off, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(mv, axis=0) / C
    diff = tl.where(mask, mv - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C
    rstd = libdevice.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = (diff * rstd * w + b).to(m_ptr.dtype.element_ty)

    xv = tl.load(x_ptr + off, mask=mask, other=0.0)
    tl.store(out_ptr + off, (xv + normed).to(out_ptr.dtype.element_ty), mask=mask)


def _post_norm_add_triton(
    m: Tensor, x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> Tensor:
    c = m.shape[-1]
    rows = m.numel() // c
    m_c = m.contiguous()
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    block_c = triton.next_power_of_2(c)
    _post_norm_add_kernel[(rows,)](
        m_c,
        x_c,
        weight.contiguous(),
        bias.contiguous(),
        out,
        float(eps),
        C=c,
        BLOCK_C=block_c,
        num_warps=4 if block_c >= 128 else 2,
    )
    return out


def _post_norm_add_eligible(
    m: Tensor, x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> bool:
    return (
        m.shape == x.shape
        and m.dtype == x.dtype
        and m.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and m.shape[-1] <= 1024
    )


ops.eloftr_post_norm_add.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_post_norm_add_eligible
)(_post_norm_add_triton)


# -- 4) Fused non-overlapping KxK depthwise conv / MaxPool2d + LayerNorm (+ optional quantize) -------------


@triton.jit
def _agg_or_pool_norm_kernel(
    x_ptr,
    agg_w_ptr,
    ln_w_ptr,
    ln_b_ptr,
    out_ptr,
    scale_ptr,
    row_scale_ptr,
    H,
    W,
    HQ,
    WQ,
    sw_c,
    sw_k,
    eps,
    K: tl.constexpr,
    C: tl.constexpr,
    PADDED_C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_POOL: tl.constexpr,
    QUANT_MODE: tl.constexpr,  # 0: dense, 1: static int8, 2: dynamic per-row int8, 3: static mxfp8
):
    """One program per output token ``(b, oy, ox)``: reduces ``K x K`` NHWC patch in registers, runs LayerNorm,
    and writes dense or quantized output."""
    pid = tl.program_id(0)
    b = (pid // (HQ * WQ)).to(tl.int64)
    rem = pid % (HQ * WQ)
    oy = rem // WQ
    ox = rem % WQ

    cols = tl.arange(0, BLOCK_C)
    c_ok = cols < C
    out_c_ok = cols < PADDED_C

    if IS_POOL:
        acc = tl.full((BLOCK_C,), -float("inf"), dtype=tl.float32)
        for dy in tl.static_range(K):
            y_row = b * H + (oy * K + dy)
            for dx in tl.static_range(K):
                x_off = (y_row * W + (ox * K + dx)) * C + cols
                xv = tl.load(x_ptr + x_off, mask=c_ok, other=-float("inf")).to(tl.float32)
                acc = tl.maximum(acc, xv)
    else:
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for dy in tl.static_range(K):
            y_row = b * H + (oy * K + dy)
            for dx in tl.static_range(K):
                x_off = (y_row * W + (ox * K + dx)) * C + cols
                xv = tl.load(x_ptr + x_off, mask=c_ok, other=0.0).to(tl.float32)
                wv = tl.load(
                    agg_w_ptr + cols.to(tl.int64) * sw_c + (dy * K + dx) * sw_k, mask=c_ok, other=0.0
                ).to(tl.float32)
                acc += xv * wv

    # Match nn.Conv2d / nn.MaxPool2d output rounding before LayerNorm.
    acc = acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    acc = tl.where(c_ok, acc, 0.0)
    mean = tl.sum(acc, axis=0) / C
    diff = tl.where(c_ok, acc - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C
    rstd = libdevice.rsqrt(var + eps)

    lw = tl.load(ln_w_ptr + cols, mask=c_ok, other=0.0).to(tl.float32)
    lb = tl.load(ln_b_ptr + cols, mask=c_ok, other=0.0).to(tl.float32)
    y = diff * rstd * lw + lb

    out_off = pid.to(tl.int64) * PADDED_C + cols
    if QUANT_MODE == 0:
        tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=out_c_ok)
    elif QUANT_MODE == 1:
        dense = y.to(tl.bfloat16).to(tl.float32)
        inv = 1.0 / tl.load(scale_ptr)
        scaled = libdevice.rint(dense * inv)
        q = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
        tl.store(out_ptr + out_off, q, mask=out_c_ok)
    elif QUANT_MODE == 2:
        dense = tl.where(c_ok, y.to(tl.bfloat16).to(tl.float32), 0.0)
        scale = tl.max(tl.abs(dense), axis=0) * (1.0 / 127.0)
        scale = tl.where(scale > 1.1920928955078125e-07, scale, 1.1920928955078125e-07)
        scaled = libdevice.rint(tl.math.div_rn(dense, scale))
        q = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
        tl.store(out_ptr + out_off, q, mask=out_c_ok)
        tl.store(row_scale_ptr + pid.to(tl.int64), scale)
    else:
        dense = tl.where(c_ok, y.to(tl.bfloat16).to(tl.float32), 0.0)
        exp = tl.load(scale_ptr + cols // 32, mask=out_c_ok, other=0).to(tl.float32)
        scaled = dense * tl.exp2(-exp)
        q = to_e4m3(tl.maximum(tl.minimum(scaled, 448.0), -448.0))
        tl.store(out_ptr + out_off, q, mask=out_c_ok)


def _launch_agg_or_pool_norm(
    x: Tensor,
    k: int,
    agg_weight: Tensor | None,
    ln_weight: Tensor,
    ln_bias: Tensor,
    eps: float,
    *,
    is_pool: bool,
    int8_scale: Tensor | None,
    int8_rows: bool,
    mxfp8_exponent: Tensor | None,
):
    b, H, W, c = x.shape
    hq, wq = H // k, W // k
    x_c = x.contiguous()
    if int8_scale is not None:
        qmode, padded_c = 1, c
        out = torch.empty((b, hq, wq, c), device=x_c.device, dtype=torch.int8)
        scale_t, row_scale = int8_scale, x_c
    elif int8_rows:
        qmode, padded_c = 2, c
        out = torch.empty((b, hq, wq, c), device=x_c.device, dtype=torch.int8)
        row_scale = torch.empty((b, hq, wq, 1), device=x_c.device, dtype=torch.float32)
        scale_t = x_c
    elif mxfp8_exponent is not None:
        qmode, padded_c = 3, padded_channels(c)
        out = torch.empty((b, hq, wq, padded_c), device=x_c.device, dtype=torch.float8_e4m3fn)
        scale_t, row_scale = mxfp8_exponent, x_c
    else:
        qmode, padded_c = 0, c
        out = torch.empty((b, hq, wq, c), device=x_c.device, dtype=x_c.dtype)
        scale_t, row_scale = x_c, x_c

    if is_pool:
        w_flat, sw_c, sw_k = x_c, 0, 0
    else:
        assert agg_weight is not None
        w_flat = agg_weight.reshape(c, k * k).contiguous()
        sw_c, sw_k = w_flat.stride(0), w_flat.stride(1)

    block_c = triton.next_power_of_2(padded_c)
    _agg_or_pool_norm_kernel[(b * hq * wq,)](
        x_c,
        w_flat,
        ln_weight.contiguous(),
        ln_bias.contiguous(),
        out,
        scale_t,
        row_scale,
        H,
        W,
        hq,
        wq,
        sw_c,
        sw_k,
        float(eps),
        K=k,
        C=c,
        PADDED_C=padded_c,
        BLOCK_C=block_c,
        IS_POOL=is_pool,
        QUANT_MODE=qmode,
        num_warps=4 if block_c >= 128 else 2,
    )
    if qmode == 1:
        return Int8Tensor(out, int8_scale)
    if qmode == 2:
        return out, row_scale
    if qmode == 3:
        return QuantizedTensor(out, mxfp8_exponent, (b, hq, wq, c))
    return out


def _agg_or_pool_eligible(
    x: Tensor,
    k: int,
    *,
    int8_scale: Tensor | None,
    int8_rows: bool,
    mxfp8_exponent: Tensor | None,
) -> bool:
    if x.ndim != 4 or x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if k not in (1, 2, 4) or x.shape[1] % k != 0 or x.shape[2] % k != 0 or x.shape[3] > 1024:
        return False
    modes = (int8_scale is not None) + bool(int8_rows) + (mxfp8_exponent is not None)
    if modes > 1:
        return False
    if modes == 1 and x.dtype != torch.bfloat16:
        return False
    return not (mxfp8_exponent is not None and not platform_for(x.device).supports(Feature.FP8_E4M3))


def _agg_norm_eligible(
    x: Tensor, agg_weight: Tensor, ln_weight: Tensor, ln_bias: Tensor, eps: float = 1e-5, *,
    int8_scale: Tensor | None = None, int8_rows: bool = False, mxfp8_exponent: Tensor | None = None,
) -> bool:
    return (
        agg_weight.ndim == 4
        and agg_weight.shape[0] == x.shape[-1]
        and agg_weight.shape[1] == 1
        and agg_weight.shape[2] == agg_weight.shape[3]
        and _agg_or_pool_eligible(
            x, agg_weight.shape[2], int8_scale=int8_scale, int8_rows=int8_rows, mxfp8_exponent=mxfp8_exponent
        )
    )


def _agg_norm_triton(
    x: Tensor, agg_weight: Tensor, ln_weight: Tensor, ln_bias: Tensor, eps: float = 1e-5, *,
    int8_scale: Tensor | None = None, int8_rows: bool = False, mxfp8_exponent: Tensor | None = None,
):
    return _launch_agg_or_pool_norm(
        x, agg_weight.shape[-1], agg_weight, ln_weight, ln_bias, eps,
        is_pool=False, int8_scale=int8_scale, int8_rows=int8_rows, mxfp8_exponent=mxfp8_exponent,
    )


ops.eloftr_agg_norm.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_agg_norm_eligible
)(_agg_norm_triton)


def _pool_norm_eligible(
    x: Tensor, pool_size: int, ln_weight: Tensor, ln_bias: Tensor, eps: float = 1e-5, *,
    int8_scale: Tensor | None = None, int8_rows: bool = False, mxfp8_exponent: Tensor | None = None,
) -> bool:
    return _agg_or_pool_eligible(
        x, pool_size, int8_scale=int8_scale, int8_rows=int8_rows, mxfp8_exponent=mxfp8_exponent
    )


def _pool_norm_triton(
    x: Tensor, pool_size: int, ln_weight: Tensor, ln_bias: Tensor, eps: float = 1e-5, *,
    int8_scale: Tensor | None = None, int8_rows: bool = False, mxfp8_exponent: Tensor | None = None,
):
    return _launch_agg_or_pool_norm(
        x, pool_size, None, ln_weight, ln_bias, eps,
        is_pool=True, int8_scale=int8_scale, int8_rows=int8_rows, mxfp8_exponent=mxfp8_exponent,
    )


ops.eloftr_pool_norm.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_pool_norm_eligible
)(_pool_norm_triton)


# -- 5) Fused LeakyReLU + static INT8 / MXFP8 quantize (FPN conv -> LeakyReLU -> conv boundaries) ---------


@triton.jit
def _leaky_relu_quantize_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    rows,
    neg_slope,
    C: tl.constexpr,
    PADDED_C: tl.constexpr,
    QUANT_MODE: tl.constexpr,  # 1: static int8, 2: static mxfp8
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    p_ok = p < rows
    c_ok = c < C
    out_c_ok = c < PADDED_C

    xv = tl.load(x_ptr + p[:, None].to(tl.int64) * C + c[None, :], mask=p_ok[:, None] & c_ok[None, :], other=0.0).to(
        tl.float32
    )
    val = tl.where(xv >= 0.0, xv, xv * neg_slope)
    out_off = p[:, None].to(tl.int64) * PADDED_C + c[None, :]
    store_mask = p_ok[:, None] & out_c_ok[None, :]
    if QUANT_MODE == 1:
        dense = val.to(tl.bfloat16).to(tl.float32)
        inv = 1.0 / tl.load(scale_ptr)
        scaled = libdevice.rint(dense * inv)
        q = tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8)
        tl.store(out_ptr + out_off, q, mask=store_mask)
    else:
        dense = tl.where(c_ok[None, :], val.to(tl.bfloat16).to(tl.float32), 0.0)
        exp = tl.load(scale_ptr + c // 32, mask=out_c_ok, other=0).to(tl.float32)
        scaled = dense * tl.exp2(-exp)[None, :]
        q = to_e4m3(tl.maximum(tl.minimum(scaled, 448.0), -448.0))
        tl.store(out_ptr + out_off, q, mask=store_mask)


def _leaky_relu_quantize_triton(
    x: Tensor,
    negative_slope: float = 0.01,
    *,
    int8_scale: Tensor | None = None,
    mxfp8_exponent: Tensor | None = None,
) -> Int8Tensor | QuantizedTensor:
    c = x.shape[-1]
    rows = x.numel() // c
    x_c = x.contiguous()
    if int8_scale is not None:
        qmode, padded_c = 1, c
        out = torch.empty(x_c.shape, device=x_c.device, dtype=torch.int8)
        scale_t = int8_scale
    else:
        assert mxfp8_exponent is not None
        qmode, padded_c = 2, padded_channels(c)
        out = torch.empty((*x_c.shape[:-1], padded_c), device=x_c.device, dtype=torch.float8_e4m3fn)
        scale_t = mxfp8_exponent

    block_p, block_c = 64, min(128, triton.next_power_of_2(padded_c))
    grid = (triton.cdiv(rows, block_p), triton.cdiv(padded_c, block_c))
    _leaky_relu_quantize_kernel[grid](
        x_c,
        out,
        scale_t,
        rows,
        float(negative_slope),
        C=c,
        PADDED_C=padded_c,
        QUANT_MODE=qmode,
        BLOCK_P=block_p,
        BLOCK_C=block_c,
        num_warps=4,
    )
    if qmode == 1:
        return Int8Tensor(out, int8_scale)
    return QuantizedTensor(out, mxfp8_exponent, tuple(x_c.shape))


def _leaky_relu_quantize_eligible(
    x: Tensor,
    negative_slope: float = 0.01,
    *,
    int8_scale: Tensor | None = None,
    mxfp8_exponent: Tensor | None = None,
) -> bool:
    if x.dtype != torch.bfloat16 or (int8_scale is None) == (mxfp8_exponent is None):
        return False
    return not (mxfp8_exponent is not None and not platform_for(x.device).supports(Feature.FP8_E4M3))


ops.eloftr_leaky_relu_quantize.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_leaky_relu_quantize_eligible
)(_leaky_relu_quantize_triton)
