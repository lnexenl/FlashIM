"""ALIKED architecture configuration: one entry per released checkpoint variant."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ALIKEDConfig:
    # Encoder channel widths (block1..4) and descriptor dimension (``dim == c4`` in every released variant).
    c1: int = 16
    c2: int = 32
    c3: int = 64
    c4: int = 128
    dim: int = 128
    # SDDH descriptor head: patch size and number of deformable sample points.
    K: int = 3
    M: int = 16
    # Encoder strides 1/2/8/32 require the input padded to a multiple of this.
    div_by: int = 32
    # DKD (differentiable keypoint detection): soft-NMS radius and soft-argmax temperature.
    dkd_radius: int = 2
    dkd_temperature: float = 0.1
    # Sparse extraction defaults: -1 selects threshold mode, >0 selects top-k mode.
    top_k: int = -1
    scores_th: float = 0.2
    n_limit: int = 5000
    # Mutual-nearest-neighbour cosine matching (``demo_pair.py``'s ``mnn_mather`` default).
    match_threshold: float = 0.75


VARIANTS: dict[str, ALIKEDConfig] = {
    "aliked-t16": ALIKEDConfig(c1=8, c2=16, c3=32, c4=64, dim=64, K=3, M=16),
    "aliked-n16": ALIKEDConfig(c1=16, c2=32, c3=64, c4=128, dim=128, K=3, M=16),
    "aliked-n16rot": ALIKEDConfig(c1=16, c2=32, c3=64, c4=128, dim=128, K=3, M=16),
    "aliked-n32": ALIKEDConfig(c1=16, c2=32, c3=64, c4=128, dim=128, K=3, M=32),
}
