"""EfficientLoFTR-specific fused ops: 2x-upsampled fine-window gathering from 1/2-res FPN features, fused bilinear
upsample + add + LeakyReLU (+ optional INT8/MXFP8 quantization), fused post-LayerNorm + residual add, fused KxK
depthwise conv / MaxPool2d + LayerNorm (+ optional quantization), and fused LeakyReLU + static quantization."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.quant.formats.int8 import INT8_MAX, quantize_symmetric, safe_scale
from flashim.quant.tensor import Int8Tensor, QuantizedTensor, quantize_int8_static, quantize_static


def _upsample_gather_windows_reference(
    feat_half: Tensor,
    ids: Tensor,
    hc: int,
    wc: int,
    window_size: int = 8,
    pad: int = 0,
) -> Tensor:
    """Bilinearly upsample ``feat_half [1, C, hc*4, wc*4]`` by 2x (``align_corners=False``) and gather only the
    matched ``window_size x window_size`` windows (stride 8, zero-padded by ``pad``) -> ``[M, window_size**2, C]``."""
    c = feat_half.shape[1]
    if ids.numel() == 0:
        return feat_half.new_empty((0, window_size * window_size, c))
    full = F.interpolate(feat_half, scale_factor=2.0, mode="bilinear", align_corners=False)
    if pad > 0:
        full = F.pad(full, (pad, pad, pad, pad))
    iy, ix = ids // wc, ids % wc
    view = full[0].unfold(1, window_size, 8).unfold(2, window_size, 8)  # [C, hc, wc, W, W]
    return view.permute(1, 2, 3, 4, 0)[iy, ix].reshape(-1, window_size * window_size, c)


eloftr_upsample_gather_windows = Op("eloftr_upsample_gather_windows", _upsample_gather_windows_reference)


def _upsample_add_leaky_relu_reference(
    x_proj: Tensor,
    m_proj: Tensor,
    negative_slope: float = 0.01,
    *,
    int8_scale: Tensor | None = None,
    mxfp8_exponent: Tensor | None = None,
) -> Tensor | Int8Tensor | QuantizedTensor:
    """``LeakyReLU(x_proj + bilinear(m_proj), negative_slope)`` on NHWC ``[B, H, W, C]`` / ``[B, h, w, C]``
    (``align_corners=False``), optionally quantized directly to static INT8 or static MXFP8."""
    _, h, w, _ = x_proj.shape
    up = F.interpolate(
        m_proj.permute(0, 3, 1, 2).float(), size=(h, w), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    y = F.leaky_relu(x_proj.float() + up, negative_slope=negative_slope).to(x_proj.dtype)
    if int8_scale is not None:
        return quantize_int8_static(y, int8_scale)
    if mxfp8_exponent is not None:
        return quantize_static(y, mxfp8_exponent)
    return y


eloftr_upsample_add_leaky_relu = Op("eloftr_upsample_add_leaky_relu", _upsample_add_leaky_relu_reference)


def _post_norm_add_reference(
    m: Tensor, x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> Tensor:
    """``x + LayerNorm(m, weight, bias, eps)`` on trailing-channel tensors ``[..., C]``."""
    normed = F.layer_norm(m.float(), (m.shape[-1],), weight.float(), bias.float(), eps).to(m.dtype)
    return (x + normed).to(x.dtype)


eloftr_post_norm_add = Op("eloftr_post_norm_add", _post_norm_add_reference)


def _quantize_output(
    y: Tensor,
    *,
    int8_scale: Tensor | None = None,
    int8_rows: bool = False,
    mxfp8_exponent: Tensor | None = None,
) -> Tensor | Int8Tensor | QuantizedTensor | tuple[Tensor, Tensor]:
    if int8_scale is not None:
        return quantize_int8_static(y, int8_scale)
    if int8_rows:
        scale = safe_scale(y.float().abs().amax(-1, keepdim=True) * (1 / INT8_MAX))
        return quantize_symmetric(y, scale), scale
    if mxfp8_exponent is not None:
        return quantize_static(y, mxfp8_exponent)
    return y


def _agg_norm_reference(
    x: Tensor,
    agg_weight: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    eps: float = 1e-5,
    *,
    int8_scale: Tensor | None = None,
    int8_rows: bool = False,
    mxfp8_exponent: Tensor | None = None,
) -> Tensor | Int8Tensor | QuantizedTensor | tuple[Tensor, Tensor]:
    """Non-overlapping KxK depthwise conv (``agg_weight [C, 1, K, K]``) + LayerNorm on NHWC ``x [B, H, W, C]``,
    optionally quantized directly to static INT8, dynamic per-row INT8, or static MXFP8."""
    k = agg_weight.shape[-1]
    c = x.shape[-1]
    pooled = F.conv2d(x.permute(0, 3, 1, 2), agg_weight, stride=k, groups=c).permute(0, 2, 3, 1)
    y = F.layer_norm(pooled.float(), (c,), ln_weight.float(), ln_bias.float(), eps).to(x.dtype)
    return _quantize_output(y, int8_scale=int8_scale, int8_rows=int8_rows, mxfp8_exponent=mxfp8_exponent)


eloftr_agg_norm = Op("eloftr_agg_norm", _agg_norm_reference)


def _pool_norm_reference(
    x: Tensor,
    pool_size: int,
    ln_weight: Tensor,
    ln_bias: Tensor,
    eps: float = 1e-5,
    *,
    int8_scale: Tensor | None = None,
    int8_rows: bool = False,
    mxfp8_exponent: Tensor | None = None,
) -> Tensor | Int8Tensor | QuantizedTensor | tuple[Tensor, Tensor]:
    """Non-overlapping KxK MaxPool2d + LayerNorm on NHWC ``x [B, H, W, C]``, optionally quantized directly to
    static INT8, dynamic per-row INT8, or static MXFP8."""
    c = x.shape[-1]
    pooled = F.max_pool2d(x.permute(0, 3, 1, 2), kernel_size=pool_size, stride=pool_size).permute(0, 2, 3, 1)
    y = F.layer_norm(pooled.float(), (c,), ln_weight.float(), ln_bias.float(), eps).to(x.dtype)
    return _quantize_output(y, int8_scale=int8_scale, int8_rows=int8_rows, mxfp8_exponent=mxfp8_exponent)


eloftr_pool_norm = Op("eloftr_pool_norm", _pool_norm_reference)


def _leaky_relu_quantize_reference(
    x: Tensor,
    negative_slope: float = 0.01,
    *,
    int8_scale: Tensor | None = None,
    mxfp8_exponent: Tensor | None = None,
) -> Int8Tensor | QuantizedTensor:
    """``LeakyReLU(x, negative_slope)`` on NHWC ``[B, H, W, C]`` fused with static INT8 or static MXFP8 quantization."""
    y = F.leaky_relu(x.float(), negative_slope=negative_slope).to(x.dtype)
    if int8_scale is not None:
        return quantize_int8_static(y, int8_scale)
    if mxfp8_exponent is not None:
        return quantize_static(y, mxfp8_exponent)
    raise ValueError("eloftr_leaky_relu_quantize requires int8_scale or mxfp8_exponent")


eloftr_leaky_relu_quantize = Op("eloftr_leaky_relu_quantize", _leaky_relu_quantize_reference)
