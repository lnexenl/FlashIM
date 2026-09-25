"""RoMaV2 refiner output heads: warp/confidence 1x1 convs + warp update + precision params.

Given the last conv block's output ``z`` ``[B, C, h, w]``:
  ``disp, delta = W_warp z + b_warp, W_conf z + b_conf``           (2 + 4 channels)
  ``warp = prev_warp + disp / (refine_init * (w, h))``
  ``delta = (d0, l00^2, l00*l10, l10^2 + l11^2)`` with ``l00 = softplus(d1) + 1e-6``,
  ``l10 = d2``, ``l11 = softplus(d3) + 1e-6``
  ``confidence = prev_confidence (zero-padded to 4) + delta``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op


def _refiner_heads_reference(
    z: Tensor, weight: Tensor, bias: Tensor, prev_warp: Tensor, prev_confidence: Tensor, refine_init: float
) -> tuple[Tensor, Tensor]:
    """``weight`` ``[6, C]`` (warp rows then confidence rows), ``bias`` ``[6]``; FP32 outputs
    ``warp`` ``[B, h, w, 2]`` and ``confidence`` ``[B, h, w, 4]``."""
    batch, channels, h, w = z.shape
    out = F.conv2d(z.float(), weight.float()[:, :, None, None], bias.float()).permute(0, 2, 3, 1)
    displacement, delta = out[..., :2], out[..., 2:]
    normalization = torch.tensor([w, h], device=z.device, dtype=prev_warp.dtype)
    warp = prev_warp + displacement / (refine_init * normalization)
    l00 = F.softplus(delta[..., 1]) + 1e-6
    l10 = delta[..., 2]
    l11 = F.softplus(delta[..., 3]) + 1e-6
    delta = torch.stack([delta[..., 0], l00 * l00, l00 * l10, l10 * l10 + l11 * l11], dim=-1)
    if prev_confidence.shape[-1] != 4:
        padded = torch.zeros_like(delta)
        padded[..., :1] = prev_confidence
        prev_confidence = padded
    return warp, prev_confidence + delta


refiner_heads = Op("refiner_heads", _refiner_heads_reference)


def _eligible(z, weight, bias, prev_warp, prev_confidence, refine_init) -> bool:
    return (
        z.ndim == 4
        and z.dtype == torch.bfloat16
        and z.is_contiguous(memory_format=torch.channels_last)
        and z.shape[1] in (32, 64, 128, 256, 512)
        and tuple(weight.shape) == (6, z.shape[1])
        and prev_confidence.shape[-1] in (1, 4)
        and z.numel() < 2**31
    )


@refiner_heads.register("triton", device=DeviceSpec("cuda"), when=_eligible)
def _refiner_heads_triton(z, weight, bias, prev_warp, prev_confidence, refine_init):
    from flashim.ops._triton.refiner import refiner_heads as kernel

    return kernel(z, weight, bias, prev_warp, prev_confidence, refine_init)
