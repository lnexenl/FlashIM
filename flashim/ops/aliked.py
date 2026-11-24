"""ALIKED-specific op: quantized deformable convolution.

Only ALIKED uses deformable convolution in FlashIM (see [[flashim-naming]]): a plain conv predicts
per-pixel sampling offsets, and a second conv is applied at those (bilinearly interpolated) positions
instead of a fixed 3x3 grid. That second conv is an ordinary GEMM once its receptive-field patches are
gathered — exactly the "im2col" a direct convolution would build — so it quantizes the same way a 1x1
conv does (:class:`~flashim.quant.methods.int8_linear.StaticInt8Conv1x1`): symmetric per-output-channel
INT8 weights, one calibrated static per-tensor INT8 activation scale. The gather itself (bilinear
interpolation at continuous, data-dependent offsets) stays FP32: it is a *position*, not a *magnitude*,
and localization error does not shrink the way GEMM rounding error does (see the model's ``quant.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.quant.tensor import Int8Tensor, quantize_int8_static


def gather_deform_patches(x: Tensor, offset: Tensor, kernel_size: int, padding: int) -> Tensor:
    """Bilinear-sampled receptive-field patches at ``offset``-shifted taps.

    DCNv1 semantics (github.com/msracver/Deformable-ConvNets, torchvision's ``deform_conv2d``): a sample
    entirely outside the image is zero; a sample straddling the border blends real and zero corners.
    Verified bit-close (<1e-6) against ``torchvision.ops.deform_conv2d`` for FP32 inputs, including offsets
    that push samples far outside the image.

    ``x``: ``[B, C, H, W]``. ``offset``: ``[B, 2 * K * K, oH, oW]``, tap-major ``(kh, kw)`` then ``(dy, dx)``
    (torchvision's layout). Returns ``[B, C, K * K, oH, oW]``, ``K * K`` flattened ``(kh, kw)`` row-major —
    matching ``weight.reshape(out_channels, C * K * K)`` for ``weight`` ``[out_channels, C, K, K]``.
    """
    b, c, h, w = x.shape
    k = kernel_size
    oh, ow = offset.shape[-2:]
    device, dtype = x.device, x.dtype

    base_h = (torch.arange(oh, device=device) - padding).unsqueeze(1) + torch.arange(k, device=device)  # [oH, K]
    base_w = (torch.arange(ow, device=device) - padding).unsqueeze(1) + torch.arange(k, device=device)  # [oW, K]
    off = offset.view(b, k, k, 2, oh, ow)
    sample_h = base_h.t()[None, :, None, :, None].to(dtype) + off[:, :, :, 0]  # [B, K, K, oH, oW]
    sample_w = base_w.t()[None, None, :, None, :].to(dtype) + off[:, :, :, 1]

    h0, w0 = sample_h.floor(), sample_w.floor()
    lh, lw = sample_h - h0, sample_w - w0
    flat = x.reshape(b, c, h * w)
    zero = torch.zeros((), device=device, dtype=dtype)

    def corner(hi: Tensor, wi: Tensor, weight: Tensor) -> Tensor:
        valid = (hi >= 0) & (hi <= h - 1) & (wi >= 0) & (wi <= w - 1)
        index = (hi.clamp(0, h - 1).long() * w + wi.clamp(0, w - 1).long()).reshape(b, 1, k * k, oh * ow)
        gathered = torch.gather(flat.unsqueeze(2).expand(-1, -1, k * k, -1), 3, index.expand(-1, c, -1, -1))
        gathered = gathered.view(b, c, k, k, oh, ow)
        return torch.where(valid.view(b, 1, k, k, oh, ow), gathered, zero) * weight.view(b, 1, k, k, oh, ow)

    patches = (
        corner(h0, w0, (1 - lh) * (1 - lw))
        + corner(h0, w0 + 1, (1 - lh) * lw)
        + corner(h0 + 1, w0, lh * (1 - lw))
        + corner(h0 + 1, w0 + 1, lh * lw)
    )
    return patches.reshape(b, c, k * k, oh, ow)


def _deform_patches_int8_reference(x: Tensor, offset: Tensor, kernel_size: int, padding: int, scale: Tensor) -> Int8Tensor:
    """Gathered patches (FP32), quantized to symmetric INT8 with a static ``scale``.

    Returns an :class:`Int8Tensor` of logical shape ``[B, oH, oW, C * K * K]`` (``matrix()`` gives the
    ``[B * oH * oW, C * K * K]`` view an INT8 GEMM against ``weight.reshape(out_channels, C * K * K)`` expects).
    """
    patches = gather_deform_patches(x.float(), offset.float(), kernel_size, padding)  # [B, C, K*K, oH, oW]
    b, c, kk, oh, ow = patches.shape
    patches = patches.permute(0, 3, 4, 1, 2).reshape(b, oh, ow, c * kk)
    return quantize_int8_static(patches, scale)


deform_patches_int8 = Op("deform_patches_int8", _deform_patches_int8_reference)


def _multiscale_fuse_reference(
    f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor, head_weight: Tensor | None = None, *, normalize: bool = True
) -> tuple[Tensor | None, Tensor | None]:
    """ALIKED's / RaCo's multi-scale feature fusion: ``fused = cat([f1, up(f2), up(f3), up(f4)])`` (bilinear,
    ``align_corners=True``, to ``f1``'s size), then ``head = selu(conv1x1(fused.float(), head_weight))`` (the score
    head's first layer, FP32; ``head_weight`` ``[8, 4C]``) and ``feature_map = F.normalize(fused, p=2, dim=1)`` in
    ``f1``'s dtype. Either output can be skipped. See ``backends/triton/aliked.py`` for the fused kernel."""
    size = f1.shape[-2:]
    upsampled = [F.interpolate(f.float(), size=size, mode="bilinear", align_corners=True) for f in (f2, f3, f4)]
    fused = torch.cat([f1.float(), *upsampled], dim=1)
    head = None if head_weight is None else F.selu(F.conv2d(fused, head_weight[:, :, None, None]))
    return head, (F.normalize(fused, p=2, dim=1).to(f1.dtype) if normalize else None)


multiscale_fuse = Op("multiscale_fuse", _multiscale_fuse_reference)


def _aliked_score_head_reference(
    head: Tensor, w1: Tensor, w2: Tensor, w3: Tensor, *, apply_sigmoid: bool = True
) -> Tensor:
    """ALIKED's ``score_head[2:]``: ``conv3x3(8->4) -> SELU -> conv3x3(4->4) -> SELU -> conv3x3(4->1)``
    (all bias-free FP32, padding=1), optionally followed by ``sigmoid``."""
    x = F.selu(F.conv2d(head.float(), w1.float(), padding=1))
    x = F.selu(F.conv2d(x, w2.float(), padding=1))
    x = F.conv2d(x, w3.float(), padding=1)
    return torch.sigmoid(x) if apply_sigmoid else x


aliked_score_head = Op("aliked_score_head", _aliked_score_head_reference)


def _aliked_sample_patches_reference(features: Tensor, corners: Tensor, size: int) -> Tensor:
    """``size x size`` integer-pixel patches of ``features`` ``[C, H, W]`` centered near ``corners`` ``[N, 2]``
    ``(x, y)`` (clamped to ``[0, W-1-size] x [0, H-1-size]``) -> ``[N, C, size, size]``."""
    c, h, w = features.shape
    if corners.shape[0] == 0:
        return features.new_empty(0, c, size, size)
    corner = (corners - size / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - size)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - size)
    offset = torch.arange(size, device=features.device)
    x, y = torch.meshgrid(offset, offset, indexing="ij")
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2).to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = features.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    return sampled.reshape(size, size, -1, c).permute(2, 3, 0, 1)


aliked_sample_patches = Op("aliked_sample_patches", _aliked_sample_patches_reference)


def _aliked_bilinear_sample_reference(features: Tensor, pos: Tensor) -> Tensor:
    """Bilinear sample (``align_corners=True``, zero padding) of ``features`` ``[C, H, W]`` at pixel coordinates
    ``pos`` ``[M, 2]`` ``(x, y)`` in ``[0, W-1] x [0, H-1]`` -> ``[M, C]`` in ``features.dtype`` (computed in FP32)."""
    from flashim.utils import device_constant

    c, h, w = features.shape
    if pos.shape[0] == 0:
        return features.new_empty(0, c)
    wh = device_constant([w - 1, h - 1], features, torch.float32)
    grid = (2.0 * pos.float() / wh - 1.0).reshape(1, pos.shape[0], 1, 2)
    out = F.grid_sample(features.float().unsqueeze(0), grid, mode="bilinear", align_corners=True)  # [1, C, M, 1]
    return out[0, :, :, 0].t().contiguous().to(features.dtype)


aliked_bilinear_sample = Op("aliked_bilinear_sample", _aliked_bilinear_sample_reference)
