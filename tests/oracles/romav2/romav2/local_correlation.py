# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/local_correlation.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

from typing import Literal

import torch
import torch.nn.functional as F
from torch.onnx import symbolic_helper

try:
    import local_corr
except ImportError:
    local_corr = None


def local_corr_wrapper(
    feature0: torch.Tensor,
    feature1: torch.Tensor,
    coords: torch.Tensor,
    local_window: torch.Tensor,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear",
    dtype=torch.float32,
):
    assert local_corr is not None
    assert padding_mode == "zeros"
    warp = (coords[..., None, :] + local_window[:, None, None]).reshape(B, h * w, K, 2)
    corr = (
        local_corr.local_corr(
            feature0.reshape(B, c, h * w).permute(0, 2, 1).float() / (c**0.5),
            feature1.permute(0, 2, 3, 1).clone().detach().float(),
            warp.clone().detach(),
            mode=sample_mode,
            normalized_coords=True,
        )
        .permute(0, 2, 1)
        .reshape(B, K, h, w)
    )
    return corr


def native_torch_local_corr(
    feature0,
    feature1,
    warp,
    local_window,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode="bilinear",
    dtype=torch.float32,
):
    corr = torch.empty((B, K, h, w), device=device, dtype=dtype)
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (
                warp[_, :, :, None] + local_window[:, None, None]
            ).reshape(1, h, w * K, 2)
            window_feature = F.grid_sample(
                feature1[_ : _ + 1],
                local_window_coords,
                padding_mode=padding_mode,
                align_corners=False,
                mode=sample_mode,
            )
            window_feature = window_feature.reshape(c, h, w, K)
        corr[_] = (
            (feature0[_, ..., None] / (c**0.5) * window_feature)
            .sum(dim=0)
            .permute(2, 0, 1)
        )
    return corr


def _fused_local_corr_onnx_symbolic(g, im_A, im_B, warp, mode):
    """TorchScript-ONNX symbolic mapping ``fused_local_corr::corr`` to the
    custom-domain node consumed by the TensorRT AOT plugin.

    Emits ``roma::local_corr`` (plugin namespace ``roma``, name ``local_corr``,
    version ``1``) with exactly the three plugin inputs: ``im_A`` [B, HW, C]
    FP32, ``im_B`` [B, H, W, C] FP32, and ``warp`` [B, HW, K, 2] FP32 pixel
    coordinates; the plugin returns [B, HW, K] FP32. Registering this symbolic
    only touches the ONNX exporter registry and never affects eager execution.
    """
    mode = symbolic_helper._maybe_get_const(mode, "s")
    if mode != "bilinear":
        raise ValueError(
            f"roma::local_corr implements bilinear sampling only, got mode={mode!r}"
        )
    corr = g.op("roma::local_corr", im_A, im_B, warp)
    warp_sizes = warp.type().sizes()
    if warp_sizes is not None and len(warp_sizes) == 4:
        corr.setType(im_A.type().with_sizes(warp_sizes[:3]))
    return corr


# Inference/export-safe: inert for eager PyTorch, picked up by torch.onnx.export
# (dynamo=False) whenever a traced graph contains fused_local_corr::corr.
torch.onnx.register_custom_op_symbolic(
    "fused_local_corr::corr", _fused_local_corr_onnx_symbolic, opset_version=11
)


def local_correlation(
    feature0: torch.Tensor,  # (B x C x H x W)
    feature1: torch.Tensor,  # (B x C x H x W)
    local_radius: int,
    warp: torch.Tensor,  # (B x H x W x 2)
    scale_factor: torch.Tensor,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear",
):
    r = local_radius
    K = (2 * r + 1) ** 2
    B, c, h, w = feature0.size()
    local_h, local_w = h, w
    device = feature0.device
    dtype = feature0.dtype
    local_window = torch.meshgrid(
        [
            torch.linspace(
                -2 * local_radius / local_h,
                2 * local_radius / local_h,
                2 * r + 1,
                device=device,
            ),
            torch.linspace(
                -2 * local_radius / local_w,
                2 * local_radius / local_w,
                2 * r + 1,
                device=device,
            ),
        ],
        indexing="ij",
    )
    local_window = (
        torch.stack((local_window[1], local_window[0]), dim=-1)[None]
        .expand(1, 2 * r + 1, 2 * r + 1, 2)
        .reshape(1, K, 2)
    )
    if local_corr is None:
        corr = native_torch_local_corr(
            feature0,
            feature1,
            warp,
            local_window,
            B,
            K,
            c,
            r,
            h,
            w,
            device,
            padding_mode,
            sample_mode,
            dtype,
        )
    else:
        corr = local_corr_wrapper(
            feature0,
            feature1,
            warp,
            local_window,
            B,
            K,
            c,
            r,
            h,
            w,
            device,
            padding_mode,
            sample_mode,
            dtype,
        )
    return corr
