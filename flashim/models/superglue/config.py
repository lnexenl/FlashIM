"""SuperGlue architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuperGlueConfig:
    """Architecture and default matching parameters of :class:`~flashim.models.superglue.SuperGlue`."""

    descriptor_dim: int = 256
    keypoint_encoder: tuple[int, ...] = (32, 64, 128, 256)
    gnn_layers: tuple[str, ...] = ("self", "cross") * 9
    num_heads: int = 4
    sinkhorn_iterations: int = 100
    match_threshold: float = 0.2
