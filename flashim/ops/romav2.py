"""RoMaV2-specific fused ops (the convolutional refiners' glue).

``romav2_refiner_input`` -- ``[f_A, warp(f_B), disp_emb, local_corr]`` in one pass.

For every pixel of A the refiner stacks, channels-last:
  ``f_a``      projected features of A                                   (P)
  ``f_ba``     ``f_b`` bilinearly sampled at the previous warp          (P)
  ``emb``      1x1 conv of the scaled displacement ``warp - grid``       (E)
  ``corr``     ``f_a / sqrt(P) . f_b`` over a (2r+1)^2 window around the
               warp, bilinear, zeros padding                             (K, optional)
The downstream conv block rounds its input to BF16, so the op returns BF16.

``romav2_refiner_heads`` -- warp/confidence 1x1 convs + warp update + precision params.

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

from flashim.ops._registry import Op
from flashim.ops.correlation import local_corr

# -- refiner input: [f_a, warp(f_b), disp_emb, local_corr] ---------------------------------------


def _refiner_input_reference(
    f_a: Tensor,
    f_b: Tensor,
    warp: Tensor,
    emb_weight: Tensor,
    emb_bias: Tensor,
    scale_factor: Tensor,
    radius: int,
    pair: int = 0,
) -> Tensor:
    """``f_a``/``f_b`` ``[B, h, w, P]`` FP32, ``warp`` ``[B, h, w, 2]``; ``emb_weight`` ``[E, 2(, 1, 1)]``;
    ``radius < 0`` disables the correlation. ``pair > 0``: ``f_a`` holds ``2*pair`` stacked images
    (both directions) and image ``i`` is matched with image ``(i + pair) % (2*pair)`` of the same
    tensor (``f_b`` is ignored). Returns channels-last BF16 ``[B, C, h, w]``.
    """
    if pair > 0:
        f_b = torch.roll(f_a, shifts=pair, dims=0)
    _, h, w, _ = f_a.shape
    f_a_nchw = f_a.permute(0, 3, 1, 2)
    f_b_nchw = f_b.permute(0, 3, 1, 2)
    f_ba = F.grid_sample(f_b_nchw, warp, mode="bilinear", align_corners=False)
    grid_1d = [torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=f_a.device) for n in (h, w)]
    gy, gx = torch.meshgrid(*grid_1d, indexing="ij")
    displacement = (warp - torch.stack((gx, gy), dim=-1)[None]).permute(0, 3, 1, 2)
    kernel = emb_weight.reshape(emb_weight.shape[0], 2)[:, :, None, None]
    emb = F.conv2d(scale_factor[None, :, None, None] * displacement, kernel, emb_bias)
    parts = [f_a_nchw, f_ba, emb]
    if radius >= 0:
        parts.append(local_corr.get("reference")(f_a_nchw, f_b_nchw, warp, radius))
    return torch.cat(parts, dim=1).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)


romav2_refiner_input = Op("romav2_refiner_input", _refiner_input_reference)


# -- refiner heads: 1x1 warp/confidence convs + warp update + precision --------------------------


def _refiner_heads_reference(
    z: Tensor, weight: Tensor, bias: Tensor, prev_warp: Tensor, prev_confidence: Tensor, refine_init: float
) -> tuple[Tensor, Tensor]:
    """``weight`` ``[6, C]`` (warp rows then confidence rows), ``bias`` ``[6]``; FP32 outputs
    ``warp`` ``[B, h, w, 2]`` and ``confidence`` ``[B, h, w, 4]``."""
    _, _, h, w = z.shape
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


romav2_refiner_heads = Op("romav2_refiner_heads", _refiner_heads_reference)
