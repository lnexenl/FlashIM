"""RoMa (v1)-specific fused ops (the convolutional refiners' input/output glue).

``roma_refiner_input`` -- ``[f_a, grid_sample(f_b, warp), disp_emb, local_corr, zero_pad]`` written once as the BF16
channels-last input of the refiner's first conv block (eliminating intermediate ``cat``, ``F.pad``, FP32->BF16 cast,
NCHW->NHWC transpose, and symmetric ``f_b`` copy).

``roma_refiner_head`` -- FP32 3-channel 1x1 output conv ``(delta_flow[2], delta_certainty[1])`` read directly from
the BF16 channels-last refiner stack output without materializing an FP32 copy of the padded activation tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.ops.correlation import local_corr
from flashim.utils import grid_sample, normalized_grid


def _refiner_input_reference(
    f_a: Tensor,
    f_b: Tensor,
    warp: Tensor,
    emb_weight: Tensor,
    emb_bias: Tensor,
    scale_factor: float,
    radius: int,
    padded_channels: int,
    pair: int = 0,
) -> Tensor:
    """``f_a`` / ``f_b`` ``[B, P, H, W]`` FP32, ``warp`` ``[B, 2, H, W]`` FP32, ``emb_weight`` ``[E, 2]``,
    ``emb_bias`` ``[E]``, ``radius < 0`` disables local correlation. ``pair > 0``: ``f_a`` holds ``2 * pair``
    stacked symmetric views and view ``i`` pairs with ``(i + pair) % (2 * pair)`` (``f_b`` ignored).
    Returns channels-last BF16 ``[B, padded_channels, H, W]`` (zero-padded beyond ``2*P + E + K``).
    """
    if pair > 0:
        f_b = torch.roll(f_a, shifts=pair, dims=0)
    b, _, h, w = f_a.shape
    warp_bhwc = warp.permute(0, 2, 3, 1)
    x_hat = grid_sample(f_b, warp_bhwc, align_corners=False, mode="bilinear")
    coords = normalized_grid(b, h, w, f_a.device).permute(0, 3, 1, 2)
    disp = (40.0 / 32.0) * scale_factor * (warp - coords)
    kernel = emb_weight.reshape(emb_weight.shape[0], 2)[:, :, None, None]
    emb = F.conv2d(disp.to(kernel.dtype), kernel, emb_bias).float()
    parts = [f_a, x_hat, emb]
    if radius >= 0:
        parts.append(local_corr.get("reference")(f_a, f_b, warp_bhwc, radius))
    d = torch.cat(parts, dim=1)
    if padded_channels > d.shape[1]:
        d = F.pad(d, (0, 0, 0, 0, 0, padded_channels - d.shape[1]))
    return d.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)


roma_refiner_input = Op("roma_refiner_input", _refiner_input_reference)


def _refiner_head_reference(
    z: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """``z`` ``[B, C_pad, H, W]`` (BF16/FP32), ``weight`` ``[3, C_orig]`` FP32, ``bias`` ``[3]`` FP32.
    Returns FP32 ``(delta_flow [B, 2, H, W], delta_certainty [B, 1, H, W])``.
    """
    c_orig = weight.shape[1]
    out = F.conv2d(z[:, :c_orig].float(), weight.float()[:, :, None, None], bias.float())
    return out[:, :-1], out[:, -1:]


roma_refiner_head = Op("roma_refiner_head", _refiner_head_reference)
