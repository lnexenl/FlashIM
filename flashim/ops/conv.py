"""Depthwise-separable conv blocks: depthwise 5x5 + bias + ReLU (+ pointwise 1x1), BF16 or
with dynamic MXFP8 activations; also ``im2col_gather`` (the byte-gather feeding quantized
``nn.Conv2d``, see ``quant/methods/conv2d.py``), ``int8_conv2d`` (its fused implicit-GEMM
replacement for a full ``groups == 1`` INT8 conv) and ``mxfp8_conv2d``, the same for static
block-scaled MXFP8."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.ops.gemm import int8_linear, mxfp8_linear, mxfp8_scaled_mm
from flashim.quant.formats.int8 import quantize_symmetric
from flashim.quant.formats.mx import quantize_dynamic_2d, swizzle_scales
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight

# -- depthwise 5x5 + bias + ReLU -> dynamic MXFP8 ----------------------------------------------


def _depthwise5x5_relu_mxfp8_reference(value: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """Depthwise 5x5 (pad 2) on BF16-rounded inputs, + bias, ReLU, dynamic MXFP8.

    ``value`` is ``[B, C, H, W]``; ``weight`` is packed ``[25, C]`` (ky, kx major);
    ``bias`` is ``[C]``. Returns E4M3 ``[B*H*W, C]`` and swizzled E8M0 scales.
    """
    channels = value.shape[1]
    kernel = weight.float().t().reshape(channels, 1, 5, 5)
    rounded = value.to(torch.bfloat16).float()
    out = F.relu(F.conv2d(rounded, kernel, bias.float(), padding=2, groups=channels))
    quantized, exponent = quantize_dynamic_2d(out.permute(0, 2, 3, 1).reshape(-1, channels))
    return quantized, swizzle_scales(exponent)


depthwise5x5_relu_mxfp8 = Op("depthwise5x5_relu_mxfp8", _depthwise5x5_relu_mxfp8_reference)


# -- depthwise 5x5 + bias + ReLU -> static INT8 ---------------------------------------------------


def _depthwise5x5_relu_int8_reference(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    """Depthwise 5x5 (pad 2) on BF16-rounded inputs, + bias, ReLU, rounded to BF16, then static INT8.

    Same operands as :func:`depthwise5x5_relu_mxfp8`; returns int8 ``[B*H*W, C]`` (``scale`` FP32).
    """
    channels = value.shape[1]
    kernel = weight.float().t().reshape(channels, 1, 5, 5)
    out = F.relu(F.conv2d(value.to(torch.bfloat16).float(), kernel, bias.float(), padding=2, groups=channels))
    return quantize_symmetric(out.to(torch.bfloat16).permute(0, 2, 3, 1).reshape(-1, channels), scale)


depthwise5x5_relu_int8 = Op("depthwise5x5_relu_int8", _depthwise5x5_relu_int8_reference)


# -- BF16 depthwise 5x5 + bias + ReLU + pointwise + bias (fused refiner block) -----------------


def _depthwise5x5_relu_pointwise_reference(
    value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor
) -> Tensor:
    """BF16 refiner conv block with FP32 accumulation.

    ``value`` ``[B, C, H, W]`` (rounded to BF16 on load); ``dw_weight`` packed ``[25, C]``;
    ``pw_weight`` ``[C_out, C_in]``; biases ``[C]``. The depthwise result is rounded to
    BF16 before the pointwise product. Returns channels-last BF16 ``[B, C, H, W]``.
    """
    channels = value.shape[1]
    kernel = dw_weight.float().t().reshape(channels, 1, 5, 5)
    hidden = F.conv2d(value.to(torch.bfloat16).float(), kernel, dw_bias.float(), padding=2, groups=channels)
    hidden = F.relu(hidden).to(torch.bfloat16).float()
    out = F.conv2d(hidden, pw_weight.float()[:, :, None, None], pw_bias.float())
    return out.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)


depthwise5x5_relu_pointwise = Op("depthwise5x5_relu_pointwise", _depthwise5x5_relu_pointwise_reference)


# -- MXFP8 depthwise-separable block: depthwise 5x5 + ReLU -> dynamic MXFP8 -> pointwise GEMM -----------


def _depthwise5x5_relu_pointwise_mxfp8_reference(
    value: Tensor,
    dw_weight: Tensor,
    dw_bias: Tensor,
    pw_weight: Tensor,
    pw_scale_swizzled: Tensor,
    pw_scale_u8: Tensor,
    pw_bias: Tensor,
) -> Tensor:
    """The unfused pipeline through normal op dispatch: ``depthwise5x5_relu_mxfp8`` then
    ``mxfp8_scaled_mm``. ``pw_weight`` ``[N, K]`` E4M3; ``pw_scale_swizzled`` is the cuBLAS layout
    of the weight exponents and ``pw_scale_u8`` the plain biased ``[N, K/32]`` bytes."""
    batch, channels, height, width = value.shape
    quantized, scale = depthwise5x5_relu_mxfp8(value, dw_weight, dw_bias)
    out = mxfp8_scaled_mm(quantized, pw_weight.t(), scale, pw_scale_swizzled, pw_bias)
    return out.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


depthwise5x5_relu_pointwise_mxfp8 = Op(
    "depthwise5x5_relu_pointwise_mxfp8", _depthwise5x5_relu_pointwise_mxfp8_reference, composite=True
)


# -- INT8 depthwise-separable block: depthwise 5x5 + ReLU -> static INT8 -> pointwise GEMM ------------


def _depthwise5x5_relu_pointwise_int8_reference(
    value: Tensor,
    dw_weight: Tensor,
    dw_bias: Tensor,
    pw_weight: Tensor,
    pw_wscale: Tensor,
    pw_bias: Tensor,
    scale: Tensor,
) -> Tensor:
    """The unfused pipeline through normal op dispatch: ``depthwise5x5_relu_int8`` then ``int8_linear``.
    ``pw_weight`` int8 ``[N, K]`` with per-row FP32 ``pw_wscale``; ``scale`` the static input scale."""
    batch, channels, height, width = value.shape
    quantized = depthwise5x5_relu_int8(value, dw_weight, dw_bias, scale)
    out = int8_linear(quantized, pw_weight, scale, pw_wscale, pw_bias)
    return out.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


depthwise5x5_relu_pointwise_int8 = Op(
    "depthwise5x5_relu_pointwise_int8", _depthwise5x5_relu_pointwise_int8_reference, composite=True
)


# -- im2col gather (quantized nn.Conv2d as im2col + GEMM) -------------------------------------------


def _im2col_gather_reference(
    x: Tensor, kh: int, kw: int, sh: int, sw: int, ph: int, pw: int, dh: int, dw: int
) -> Tensor:
    """``x`` ``[B, H, W, C]`` of a 1-byte dtype -> ``[B*Ho*Wo, kh*kw*C]`` in ``(ky, kx, c)`` order (zero padding).

    The reference pads once (``F.pad``) and concatenates each tap's strided view (``torch.cat``); both are full
    extra passes over the data. The Triton impl gathers all taps of a row in one kernel, reading each input byte
    once per tap directly (no padded copy, no concat) — see ``backends/triton/conv.py``.
    """
    b, h, w, _ = x.shape
    raw = x.view(torch.uint8)
    if ph or pw:
        raw = F.pad(raw, (0, 0, pw, pw, ph, ph))  # zero byte == +0 in int8 and E4M3
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    taps = [
        raw[:, ky * dh : ky * dh + (ho - 1) * sh + 1 : sh, kx * dw : kx * dw + (wo - 1) * sw + 1 : sw]
        for ky in range(kh)
        for kx in range(kw)
    ]
    cols = taps[0] if len(taps) == 1 else torch.cat(taps, dim=-1)
    return cols.reshape(b * ho * wo, -1).view(x.dtype)


im2col_gather = Op("im2col_gather", _im2col_gather_reference)


# -- INT8 nn.Conv2d (groups == 1) as one fused implicit GEMM ------------------------------------


def _int8_conv2d_reference(
    x: Tensor, weight: Tensor, a_scale: Tensor, w_scale: Tensor, bias: Tensor | None,
    kh: int, kw: int, sh: int, sw: int, ph: int, pw: int, dh: int, dw: int,
) -> Tensor:
    """``x`` int8 ``[B, H, W, Cin]``, ``weight`` int8 ``[Cout, kh*kw*Cin]`` in ``(ky, kx, c)`` order (as
    ``StaticInt8Conv2d.from_conv`` lays it out) -> BF16 ``[B, Ho, Wo, Cout]``.

    Reference is the unfused pipeline (``im2col_gather`` then ``int8_linear``) ``StaticInt8Conv2d`` used
    before backends fused the gather into the GEMM's K-loop: one tap per full-``Cin`` ``tl.dot``, gathered
    straight from ``x`` with no materialized im2col buffer — see ``backends/triton/conv.py``.
    """
    b, h, w, _ = x.shape
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    cols = im2col_gather(x, kh, kw, sh, sw, ph, pw, dh, dw)
    out = int8_linear(cols, weight, a_scale, w_scale, bias)
    return out.reshape(b, ho, wo, -1)


int8_conv2d = Op("int8_conv2d", _int8_conv2d_reference, composite=True)


# -- static MXFP8 nn.Conv2d (groups == 1) as one fused implicit GEMM ---------------------------


def fused_mxfp8_conv2d_worthwhile(taps: int, channel_blocks: int) -> bool:
    """Whether the fused implicit-GEMM (below) is worth it for this shape.

    Every tap needs its own K-loop pass over the 32-channel blocks (block scales aren't shared across taps the way
    INT8's per-tensor scale is), so a tap costs ``channel_blocks`` small-K ``tl.dot`` calls instead of
    ``int8_conv2d``'s one full-``Cin`` dot; total small-K dot calls = ``taps * channel_blocks``, and *either*
    factor alone can make that too many to keep tensor cores fed. Measured wins (36 dot calls or fewer): ALIKE
    3x3, Cin <= 128, including its full-resolution ``block1.conv2`` (9 dot calls at ~1.9M rows -- row count is
    *not* a factor here, only dot-call count: fp8 eager 1600px 24.4ms -> 14.4ms), SuperPoint 3x3 Cin <= 128
    (36 dot calls, including a full-resolution stem conv at ~30x the rows of its other quantized convs; 19.9ms ->
    12.0ms), XFeat 1x1/3x3 (<= 16 dot calls; 6.4ms -> 3.3ms). Measured regressions (above 36): DISK 5x5 Cin <= 128
    (100 dot calls; 66.4ms -> 86.1ms), RIPE's VGG backbone, 3x3 up to Cin = 512 (144 dot calls; 63.9ms -> 110.3ms
    -- a plain tap-count gate missed this one since the blowup came from channel count, not taps), and LoFTR's
    ResNet-FPN backbone, 3x3 Cin up to 224 (63 dot calls; semi-dense match 46.7ms bf16 -> 124.6ms with only its
    <= 36 layers excluded, 50.9ms once the 63-dot-call ones were too). A row-count term was tried and dropped: it
    excluded ALIKE's and SuperPoint's full-resolution 9/36-dot-call convs (which are fine despite huge row counts)
    without changing the outcome for any of the regressions above, which were already excluded on dot-call count
    alone. The ceiling is pinned to the highest value with two independent confirmations (ALIKE and SuperPoint,
    both 36) rather than interpolating from one shape's regression; revisit with a genuine intermediate data
    point (a model between 36 and 63) rather than nudging this constant on further model-specific failures. Both
    :mod:`backends.triton.conv` (the Triton impl's ``when``) and ``StaticMXFP8Conv2d`` (which forward path to
    take for the *unfused* case -- see its docstring for why that must never fall through to the op's reference
    fallback) call this, so the two never drift apart.
    """
    return taps * channel_blocks <= 36


def _mxfp8_conv2d_reference(
    x: Tensor, channel_exponent: Tensor, weight: Tensor, weight_exponent: Tensor, bias: Tensor | None,
    kh: int, kw: int, sh: int, sw: int, ph: int, pw: int, dh: int, dw: int,
) -> Tensor:
    """``x`` E4M3 ``[B, H, W, Cin]`` (``Cin`` already padded to a multiple of 32), ``channel_exponent`` int8
    ``[Cin / 32]`` (one static exponent per input-channel block, the *same* for every tap: a tap only changes which
    weight reads a pixel, not the pixel's own scale). ``weight`` E4M3 ``[Cout, kh*kw*Cin]`` in ``(ky, kx, c)`` order
    with ``weight_exponent`` int8 ``[Cout, kh*kw*Cin / 32]`` (as ``StaticMXFP8Conv2d.from_conv`` lays it out) ->
    BF16 ``[B, Ho, Wo, Cout]``.

    Reference is the unfused pipeline (``im2col_gather`` then ``mxfp8_linear``) ``StaticMXFP8Conv2d`` used before:
    the gather gives every tap its own (duplicated) block of ``channel_exponent`` so the flattened K axis has one
    exponent per (tap, channel block), matching ``mxfp8_linear``'s convention; the ``triton`` impl instead gathers
    per tap in the GEMM's K-loop with no materialized im2col buffer, reading the un-duplicated ``channel_exponent``
    once per tap — see ``backends/triton/conv.py``.
    """
    b, h, w, _ = x.shape
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    taps = kh * kw
    cols = im2col_gather(x, kh, kw, sh, sw, ph, pw, dh, dw)
    activation = QuantizedTensor(cols, channel_exponent.repeat(taps), (cols.shape[0], cols.shape[1]))
    packed_weight = QuantizedWeight(weight, weight_exponent, tuple(weight.shape))
    out = mxfp8_linear(activation, bias, packed_weight)
    return out.reshape(b, ho, wo, -1)


mxfp8_conv2d = Op("mxfp8_conv2d", _mxfp8_conv2d_reference, composite=True)
