"""Architecture configuration for MASt3R (CroCo ViT-L encoder, ViT-B cross-attention decoder, CatMLP+DPT heads)."""

from __future__ import annotations

from dataclasses import dataclass

_INF = float("inf")


@dataclass(frozen=True)
class MASt3RConfig:
    """Parameters of ``AsymmetricMASt3R`` (defaults match ``MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric``)."""

    img_size: tuple[int, int] = (512, 512)
    patch_size: int = 16
    enc_embed_dim: int = 1024
    enc_depth: int = 24
    enc_num_heads: int = 16
    dec_embed_dim: int = 768
    dec_depth: int = 12
    dec_num_heads: int = 12
    mlp_ratio: int = 4
    norm_eps: float = 1e-6
    norm_im2_in_dec: bool = True
    rope_freq: float = 100.0
    local_feat_dim: int = 24
    hidden_dim_factor: float = 4.0
    dpt_feature_dim: int = 256
    depth_mode: tuple[str, float, float] = ("exp", -_INF, _INF)
    conf_mode: tuple[str, float, float] = ("exp", 1.0, _INF)
    desc_mode: str = "norm"
    two_confs: bool = True
    desc_conf_mode: tuple[str, float, float] = ("exp", 0.0, _INF)
    # Default inference matching parameters
    resize: int = 512
    subsample: int = 8
    border: int = 3
    square_ok: bool = False
