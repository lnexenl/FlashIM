"""Normalization + activation ops: fused InstanceNorm2d + PReLU (DISK) and RGB-mean + InstanceNorm2d (+ unfold, XFeat)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op


def _instance_norm_prelu_reference(
    x: Tensor, weight: Tensor, eps: float = 1e-5, *, out_channels_last: bool | None = None
) -> Tensor:
    """``F.prelu(F.instance_norm(x, eps=eps), weight)`` for ``x`` ``[B, C, H, W]`` (FP32 / BF16 / FP16) and
    ``weight`` ``[C]``. ``out_channels_last``: ``None`` preserves ``x``'s memory format, ``True`` returns NHWC,
    ``False`` returns contiguous NCHW."""
    out = F.prelu(F.instance_norm(x.float(), eps=eps).to(x.dtype), weight.to(x.dtype))
    if out_channels_last is True:
        return out.contiguous(memory_format=torch.channels_last)
    if out_channels_last is False:
        return out.contiguous()
    return out


instance_norm_prelu = Op("instance_norm_prelu", _instance_norm_prelu_reference)


def _rgb_mean_instance_norm_reference(
    image: Tensor, eps: float = 1e-5, *, dtype: torch.dtype | None = None, unfold: int = 0
) -> Tensor | tuple[Tensor, Tensor]:
    """``F.instance_norm(image.mean(dim=1, keepdim=True), eps=eps).to(dtype)`` for ``image`` ``[B, 3, H, W]``.
    When ``unfold > 0`` (e.g. ``8`` for XFeat's keypoint head), also returns ``pixel_unshuffle(gray, unfold)``
    (``[B, unfold**2, H // unfold, W // unfold]``)."""
    out_dtype = image.dtype if dtype is None else dtype
    gray = F.instance_norm(image.mean(dim=1, keepdim=True).float(), eps=eps).to(out_dtype)
    if unfold > 0:
        b, c, h, w = gray.shape
        u = gray.unfold(2, unfold, unfold).unfold(3, unfold, unfold).reshape(b, c, h // unfold, w // unfold, unfold**2)
        unfolded = u.permute(0, 1, 4, 2, 3).reshape(b, -1, h // unfold, w // unfold)
        return gray, unfolded
    return gray


rgb_mean_instance_norm = Op("rgb_mean_instance_norm", _rgb_mean_instance_norm_reference)
