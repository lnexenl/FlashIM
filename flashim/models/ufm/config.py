"""Architecture and inference configuration for UFM (UniFlowMatch)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UFMConfig:
    """``UFM-Base`` (``UniFlowMatchConfidence``): DINOv2 ViT-L/14 encoder, 12-layer global-attention
    multi-view transformer and two DPT regression heads (dense flow + covisibility mask).

    Default working resolution is upstream's ``(420, 560)`` (``(H, W)``; upstream stores ``(W, H) = (560, 420)``).
    """

    # DINOv2 ViT-L/14 encoder
    encoder_dim: int = 1024
    encoder_depth: int = 24
    encoder_heads: int = 16
    patch_size: int = 14
    pos_embed_size: int = 518
    # Multi-view global-attention transformer
    info_dim: int = 768
    info_depth: int = 12
    info_heads: int = 12
    info_indices: tuple[int, ...] = (5, 8)
    # DPT heads (flow and covisibility)
    dpt_input_dims: tuple[int, ...] = (1024, 768, 768, 768)
    dpt_layer_dims: tuple[int, ...] = (96, 192, 384, 768)
    dpt_feature_dim: int = 256
    dpt_hidden_dims: tuple[int, int] = (128, 128)
    flow_base_shape: tuple[int, int] = (224, 224)
    # Default working resolution (H, W)
    height: int = 420
    width: int = 560
