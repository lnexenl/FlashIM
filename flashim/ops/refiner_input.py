"""RoMaV2 refiner input assembly: ``[f_A, warp(f_B), disp_emb, local_corr]`` in one pass.

For every pixel of A the refiner stacks, channels-last:
  ``f_a``      projected features of A                                   (P)
  ``f_ba``     ``f_b`` bilinearly sampled at the previous warp          (P)
  ``emb``      1x1 conv of the scaled displacement ``warp - grid``       (E)
  ``corr``     ``f_a / sqrt(P) . f_b`` over a (2r+1)^2 window around the
               warp, bilinear, zeros padding                             (K, optional)
The downstream conv block rounds its input to BF16, so the op returns BF16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op
from flashim.ops.local_corr import local_corr


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
    batch, h, w, _ = f_a.shape
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


refiner_input = Op("refiner_input", _refiner_input_reference)


def _eligible(f_a, f_b, warp, emb_weight, emb_bias, scale_factor, radius, pair=0) -> bool:
    return (
        f_a.dtype == f_b.dtype == warp.dtype == torch.float32
        and f_a.is_contiguous()
        and f_b.is_contiguous()
        and warp.is_contiguous()
        and f_a.shape == f_b.shape
        and f_a.shape[-1] <= 256
        and emb_weight.shape[0] <= 128
        and radius <= 3
        and f_a.numel() * 4 < 2**31
    )


@refiner_input.register("triton", device=DeviceSpec("cuda"), when=_eligible)
def _refiner_input_triton(f_a, f_b, warp, emb_weight, emb_bias, scale_factor, radius, pair=0):
    from flashim.ops._triton.refiner import refiner_input as kernel

    return kernel(f_a, f_a if pair > 0 else f_b, warp, emb_weight, emb_bias, scale_factor, radius, pair)
