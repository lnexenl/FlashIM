"""SDDH: sparse deformable descriptor head (module names follow the released checkpoints)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sample_patches(features: Tensor, corners: Tensor, size: int) -> Tensor:
    """``size x size`` integer-pixel patches of ``features`` ``[C, H, W]`` centered near ``corners`` ``[N, 2]``
    ``(x, y)``, clamped to the image. Dependency-free port of LightGlue's ``get_patches`` fallback
    (github.com/cvg/LightGlue), used instead of ALIKED's optional ``custom_ops`` CUDA extension."""
    c, h, w = features.shape
    corner = (corners - size / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - size)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - size)
    offset = torch.arange(size, device=features.device)
    x, y = torch.meshgrid(offset, offset, indexing="ij")
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2).to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = features.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    return sampled.reshape(size, size, -1, c).permute(2, 3, 0, 1)


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
        wh = features.new_tensor([w - 1, h - 1])
        max_offset = max(h, w) / 4.0
        num_kpts = keypoints.shape[0]

        kpts_wh = (keypoints / 2 + 0.5) * wh
        patch = sample_patches(features, kpts_wh.long(), self.kernel_size)  # [N, C, K, K]
        offset = self.offset_conv(patch)[:, :, 0, 0].clamp(-max_offset, max_offset)
        offset = offset.view(num_kpts, 2, self.n_pos).permute(0, 2, 1)  # [N, n_pos, 2]

        pos = kpts_wh.unsqueeze(1) + offset  # [N, n_pos, 2]
        pos = (2.0 * pos / wh[None] - 1).reshape(1, num_kpts * self.n_pos, 1, 2)
        sampled = F.grid_sample(features.unsqueeze(0), pos, mode="bilinear", align_corners=True)  # [1, C, N*n_pos, 1]
        sampled = sampled.reshape(c, num_kpts, self.n_pos, 1).permute(1, 0, 2, 3)  # [N, C, n_pos, 1]

        sampled = torch.selu(self.sf_conv(sampled)).squeeze(-1)  # [N, C, n_pos]
        descriptors = torch.einsum("ncp,pcd->nd", sampled, self.agg_weights)
        return F.normalize(descriptors, p=2.0, dim=1)
