"""SDDH: sparse deformable descriptor head (module names follow the released checkpoints)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.utils import device_constant


def sample_patches(features: Tensor, corners: Tensor, size: int) -> Tensor:
    """``size x size`` integer-pixel patches of ``features`` ``[C, H, W]`` centered near ``corners`` ``[N, 2]``
    ``(x, y)``, clamped to the image."""
    return ops.aliked_sample_patches(features, corners, size)


class SDDH(nn.Module):
    """Per-keypoint descriptor: a local patch predicts ``n_pos`` deformable sample offsets around the
    keypoint; the sampled features are mixed by a learned per-position weight (``agg_weights``).

    Only the released variants' configuration is supported: no sampling mask, no ``conv2D`` mixing (both
    upstream options are unused by every published checkpoint).
    """

    def __init__(self, dim: int, kernel_size: int, n_pos: int, gate: nn.Module) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_pos = n_pos
        channels = 2 * n_pos
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dim, channels, kernel_size=kernel_size, bias=True),
            gate,
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.sf_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.agg_weights = nn.Parameter(torch.rand(n_pos, dim, dim))

    def forward(self, features: Tensor, keypoints: Tensor) -> Tensor:
        """``features``: ``[C, H, W]``; ``keypoints``: ``[N, 2]`` normalized ``(x, y)`` in ``[-1, 1]``.
        Returns L2-normalized descriptors ``[N, C]``."""
        c, h, w = features.shape
        keypoints = keypoints.to(features.dtype)  # keeps the sampling grid in `features`' dtype for grid_sample
        wh = device_constant([w - 1, h - 1], features)
        max_offset = max(h, w) / 4.0
        num_kpts = keypoints.shape[0]

        kpts_wh = (keypoints / 2 + 0.5) * wh
        patch = sample_patches(features, kpts_wh.long(), self.kernel_size)  # [N, C, K, K]
        c0, gate, c2 = self.offset_conv[0], self.offset_conv[1], self.offset_conv[2]
        if isinstance(c0, nn.Conv2d) and isinstance(c2, nn.Conv2d):
            hidden = gate(F.linear(patch.reshape(num_kpts, -1), c0.weight.reshape(c0.out_channels, -1), c0.bias))
            offset = F.linear(hidden, c2.weight.reshape(c2.out_channels, -1), c2.bias).clamp(-max_offset, max_offset)
        else:
            offset = self.offset_conv(patch)[:, :, 0, 0].clamp(-max_offset, max_offset)
        offset = offset.view(num_kpts, 2, self.n_pos).permute(0, 2, 1)  # [N, n_pos, 2]

        pos = kpts_wh.unsqueeze(1) + offset  # [N, n_pos, 2] in pixel coordinates
        sampled = ops.aliked_bilinear_sample(features, pos.reshape(-1, 2)).reshape(num_kpts, self.n_pos, c)
        sampled = torch.selu(F.linear(sampled, self.sf_conv.weight.reshape(c, c)))  # [N, n_pos, C]
        descriptors = torch.einsum("npc,pcd->nd", sampled, self.agg_weights)
        return F.normalize(descriptors, p=2.0, dim=1)
