"""TraqPoint architecture configuration (DINOv3 ConvNeXt-B + 4-level deformable transformer + RL detector)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraqPointConfig:
    # DINOv3 ConvNeXt-Base backbone (strides 4, 8, 16, 32).
    backbone_depths: tuple[int, ...] = (3, 3, 27, 3)
    backbone_dims: tuple[int, ...] = (128, 256, 512, 1024)
    # Multi-scale deformable transformer over `num_levels` feature levels (ReLU FFN).
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ffn: int = 1024
    num_levels: int = 4
    num_points: int = 8
    # Keypoint detector: small conv net on the unnormalized [0, 1] image.
    detector_dims: tuple[int, ...] = (8, 16, 32, 64)
    # Sparse extraction defaults (upstream configs/default.yaml).
    top_k: int = 4096
    nms_radius: int = 2
    softargmax_temperature: float = 0.1
    detection_threshold: float = 0.0
    match_inv_temperature: float = 20.0
    match_threshold: float = 0.01
