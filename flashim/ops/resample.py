"""Resampling: bilinear upsampling (``align_corners=False``) of a coarse map joined to a full-resolution one by
concatenation or addition, in one pass (multi-scale decoders: DeDoDe / DaD / RIPE); PIL's bicubic image resize
(8-bit rounding after each separable pass) to one or several sizes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.quant.tensor import Int8Tensor, quantize_int8_static


def _upsample_concat_reference(features: Tensor, coarse: Tensor, *, coarse_first: bool = False) -> Tensor:
    """``cat((features, upsample(coarse)), 1)`` (or ``cat((upsample(coarse), features), 1)`` when ``coarse_first``):
    ``coarse`` ``[B, Cc, h, w]`` upsampled bilinearly (``align_corners=False``, in FP32) to ``features``'
    ``[B, Cf, H, W]`` size and cast to its dtype -> ``[B, Cf + Cc, H, W]``."""
    upsampled = F.interpolate(coarse.float(), size=features.shape[-2:], mode="bilinear", align_corners=False).to(features.dtype)
    return torch.cat((upsampled, features), dim=1) if coarse_first else torch.cat((features, upsampled), dim=1)


upsample_concat = Op("upsample_concat", _upsample_concat_reference)


def _upsample_concat_int8_reference(features: Tensor, coarse: Tensor, scale: Tensor) -> Int8Tensor:
    """:func:`upsample_concat` (BF16) statically quantized for the INT8 1x1 convolution it feeds: ``round(x / scale)``
    saturated to +-127 -> :class:`Int8Tensor` ``[B, H, W, Cf + Cc]`` (channels-last, the convolution's GEMM rows)."""
    dense = _upsample_concat_reference(features.to(torch.bfloat16), coarse)
    return quantize_int8_static(dense.permute(0, 2, 3, 1), scale)


upsample_concat_int8 = Op("upsample_concat_int8", _upsample_concat_int8_reference)


def _upsample_add_reference(coarse: Tensor, delta: Tensor) -> Tensor:
    """``upsample(coarse) + delta`` in FP32: ``coarse`` ``[B, C, h, w]`` upsampled bilinearly (``align_corners=False``) to
    ``delta``'s ``[B, C, H, W]`` size -> FP32."""
    upsampled = F.interpolate(coarse.float(), size=delta.shape[-2:], mode="bilinear", align_corners=False)
    return upsampled + delta.float()


upsample_add = Op("upsample_add", _upsample_add_reference)


def _quantize8(image: Tensor) -> Tensor:
    return (image * 255).round().clamp(0, 255) / 255


def _pil_bicubic_resize_reference(image: Tensor, sizes: Sequence[tuple[int, int]]) -> tuple[Tensor, ...]:
    """``PIL.Image.resize((w, h))`` (BICUBIC) of an 8-bit image to each ``(h, w)`` of ``sizes``, in torch: PIL
    resamples width first, then height, with an antialiased bicubic (a = -0.5, torch's ``antialias=True`` kernel) and
    rounds to 8 bits after each pass (a side already at its size is not resampled). ``image`` ``[B, C, H, W]``: uint8
    (converted as ``load_image`` does, ``x.float() / 255``) or float in [0, 1] -> FP32 ``[B, C, h, w]`` per size.
    Measured against Pillow 12.3 on two MegaDepth / toronto images, 1024 px and 784 x 784, CPU and CUDA: at most 1/255
    apart, >= 99.95% of the values equal (one bicubic pass without the rounding: up to 6/255, 86% equal; bilinear:
    31/255)."""
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    out = []
    for height, width in sizes:
        x = image
        if x.shape[-1] != width:
            x = _quantize8(F.interpolate(x, size=(x.shape[-2], width), mode="bicubic", antialias=True))
        if x.shape[-2] != height:
            x = _quantize8(F.interpolate(x, size=(height, width), mode="bicubic", antialias=True))
        out.append(x)
    return tuple(out)


pil_bicubic_resize = Op("pil_bicubic_resize", _pil_bicubic_resize_reference)


def _multiscale_score_reference(
    f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    """4-scale bilinear fusion (``align_corners=True`` to ``f1``'s size) + 1-channel FP32 1x1 projection:
    ``conv1x1(cat([f1, up(f2), up(f3), up(f4)], 1).float(), weight, bias)`` -> ``[B, 1, H, W]`` FP32.
    ``weight`` is ``[4 * C]`` FP32, ``bias`` is ``[1]`` (or scalar) FP32 or ``None``."""
    size = f1.shape[-2:]
    upsampled = [F.interpolate(f.float(), size=size, mode="bilinear", align_corners=True) for f in (f2, f3, f4)]
    fused = torch.cat([f1.float(), *upsampled], dim=1)
    w = weight.float().reshape(1, -1, 1, 1)
    b = None if bias is None else bias.float().reshape(1)
    return F.conv2d(fused, w, b)


multiscale_score = Op("multiscale_score", _multiscale_score_reference)
