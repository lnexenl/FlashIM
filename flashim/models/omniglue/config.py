"""Architecture configuration for OmniGlue (Jiang et al., CVPR 2024)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OmniGlueConfig:
    """Architecture hyperparameters matching Google Research's released ``og_export`` SavedModel."""

    descriptor_dim: int = 256
    keypoint_encoder: tuple[int, ...] = (32, 64, 128, 256)
    num_heads: int = 4
    num_layers: int = 9  # 9 alternating (self, cross) pairs = 18 attentional aggregation layers
    sinkhorn_iterations: int = 100
    match_threshold: float = 1e-3  # OmniGlue.FindMatches default (demo.py filters further at 0.02)
    dino_dim: int = 768
    dino_max_size: int = 630
    dino_patch_size: int = 14
    dino_depth: int = 12
    dino_heads: int = 12
    bn_eps: float = 1e-5
