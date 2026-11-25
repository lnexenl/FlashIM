"""XoFTR architecture and inference configuration (defaults match upstream's 640 px inference config)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XoFTRConfig:
    # ResNet_8_2 backbone (1/2, 1/4, 1/8 feature maps, no FPN top-down path).
    initial_dim: int = 128
    block_dims: tuple[int, int, int] = (128, 196, 256)
    # Coarse 1/8 LoFTR linear-attention transformer.
    d_model: int = 256
    coarse_nhead: int = 8
    coarse_layer_names: tuple[str, ...] = ("self", "cross") * 4
    # Coarse matching (union of one-way softmax thresholded nearest neighbours).
    coarse_threshold: float = 0.3
    coarse_temperature: float = 0.1
    border_rm: int = 2
    # Hierarchical medium (1/4, 3x3) -> fine (1/2, 5x5) window attention.
    medium_window_size: int = 3
    fine_window_size: int = 5
    nhead_medium_level: int = 7
    nhead_fine_level: int = 8
    mlp_hidden_dim_coef: int = 2
    # Fine matching (dual-softmax in the 5x5 window) + sub-pixel MLP refinement.
    fine_threshold: float = 0.1
    fine_temperature: float = 0.1
    denser: bool = False
    # Default working long side (upstream demo / VisTIR evaluation protocol).
    default_resize: int = 640
