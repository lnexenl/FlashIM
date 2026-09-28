"""``local_corr`` through the ``fused-local-corr`` CUDA extension; loaded only when the package is installed."""

from __future__ import annotations

import local_corr as lc
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.ops.correlation import local_window


@ops.local_corr.register("fused_local_corr", device=DeviceSpec("cuda"))
def _local_corr_fused(feature0: Tensor, feature1: Tensor, warp: Tensor, radius: int) -> Tensor:
    batch, channels, h, w = feature0.shape
    window = local_window(radius, h, w, feature0.device)
    k = window.shape[1]
    coords = (warp[..., None, :] + window[:, None, None]).reshape(batch, h * w, k, 2)
    corr = lc.local_corr(
        feature0.reshape(batch, channels, h * w).permute(0, 2, 1).float() / channels**0.5,
        feature1.permute(0, 2, 3, 1).clone().float(),
        coords.clone(),
        mode="bilinear",
        normalized_coords=True,
    )
    return corr.permute(0, 2, 1).reshape(batch, k, h, w)
