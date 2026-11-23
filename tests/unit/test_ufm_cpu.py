"""CPU unit tests for UFM (synthetic tiny config; no checkpoint download needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import flashim
from flashim.layers.transformer import TransformerBlock
from flashim.models.ufm import UFM, UFMConfig


def _tiny_model() -> UFM:
    cfg = UFMConfig(
        encoder_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        patch_size=14,
        pos_embed_size=56,
        info_dim=48,
        info_depth=2,
        info_heads=4,
        info_indices=(0,),
        dpt_input_dims=(64, 48, 48, 48),
        dpt_layer_dims=(16, 32, 48, 48),
        dpt_feature_dim=32,
        dpt_hidden_dims=(16, 16),
        height=56,
        width=56,
    )
    # Two intermediate indices needed for the 4 DPT inputs [feat1, inter0, inter1, final]
    cfg = UFMConfig(**{**cfg.__dict__, "info_depth": 3, "info_indices": (0, 1)})
    return UFM(cfg).eval()


def test_tiny_forward_and_match_cpu():
    model = _tiny_model()
    blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
    assert len(blocks) == model.config.encoder_depth + model.config.info_depth

    img_a = np.full((80, 100, 3), 128, dtype=np.uint8)
    img_b = np.full((90, 120, 3), 120, dtype=np.uint8)

    flow, covis = model.predict_flow(img_a, img_b)
    assert flow.shape == (1, 2, 80, 100) and flow.dtype == torch.float32
    assert covis.shape == (1, 80, 100) and covis.min() >= 0.0 and covis.max() <= 1.0

    gen = torch.Generator(device="cpu").manual_seed(0)
    matches = model.match(img_a, img_b, max_matches=64, generator=gen)
    assert matches.keypoints_A.shape == matches.keypoints_B.shape
    assert matches.keypoints_A.shape[1] == 2 and len(matches) <= 64
    assert matches.confidence.shape == (len(matches),)
    assert matches.dense is not None and matches.dense.warp_AB.shape == (1, 56, 56, 2)

    # Custom resize (rounded to multiple of 14) and threshold
    m_resized = model.match(img_a, img_b, resize=70, max_matches=32, threshold=0.0, generator=torch.Generator().manual_seed(0))
    assert m_resized.dense.warp_AB.shape[1] % 14 == 0 and m_resized.dense.warp_AB.shape[2] % 14 == 0


def test_ufm_spec_resolves():
    for preset in ("fp32", "bf16", "int8", "fp8"):
        cfg = flashim.resolve("ufm", preset)
        assert cfg.model == "ufm" and cfg.preset == preset
    with pytest.raises(ValueError, match="max_matches"):
        _tiny_model().match(np.zeros((56, 56, 3), dtype=np.uint8), np.zeros((56, 56, 3), dtype=np.uint8), max_matches=0)
