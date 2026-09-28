"""RDD architecture configuration (the released ConvNeXt-tiny model is the default)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RDDConfig:
    # ConvNeXt backbone (strides 4, 8, 16, 32).
    backbone_depths: tuple[int, ...] = (3, 3, 9, 3)
    backbone_dims: tuple[int, ...] = (96, 192, 384, 768)
    # Deformable transformer over `num_levels` feature levels (the extra level is a stride-2 conv).
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ffn: int = 1024
    num_levels: int = 5
    num_points: int = 8
    # Keypoint detector: small conv net on the image ("legacy" detector of the release).
    detector_dims: tuple[int, ...] = (8, 16, 32, 64)
    # Sparse extraction defaults.
    top_k: int = 4096
    nms_radius: int = 2
    softargmax_temperature: float = 0.1
    match_inv_temperature: float = 20.0
    match_threshold: float = 0.01
