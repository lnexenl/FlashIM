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


def test_rotate_ufm_preserves_fp32_output_cpu():
    from flashim.models.ufm import quant
    from flashim.quant.passes.transformer_block import fold_layer_scale

    torch.manual_seed(0)
    # head_dim = 64/4 = 16 (encoder) and 48/3 = 16 (info_sharing), matching real UFM's power-of-two head_dim=64.
    base = _tiny_model().config
    model = UFM(UFMConfig(**{**base.__dict__, "info_heads": 3})).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.weight.data.uniform_(0.5, 2.0)
            m.bias.data.normal_(0, 0.2)
    a = torch.randn(1, 3, 56, 56)
    b = torch.randn(1, 3, 56, 56)
    with torch.inference_mode():
        f_ref, cm_ref, cl_ref = model(a, b)
        fold_layer_scale(model)
        n_rot = quant.rotate_ufm(model)
        assert n_rot > 0
        f_rot, cm_rot, cl_rot = model(a, b)
    torch.testing.assert_close(f_rot, f_ref, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(cm_rot, cm_ref, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(cl_rot, cl_ref, atol=2e-4, rtol=1e-4)


def test_ufm_quant_configs_respect_protected():
    from flashim.models.ufm import quant

    full_names = [
        "encoder.model.patch_embed.proj",
        "info_sharing.proj_embed",
        "head1.0.1.0",
        "uncertainty_head.0.1.0",
        *(f"encoder.model.blocks.{i}.{s}" for i in range(24) for s in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")),
        *(
            f"info_sharing.self_attention_blocks.{i}.{s}"
            for i in range(12)
            for s in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
        ),
    ]
    for cfg in (quant.INT8_CONFIG, quant.FP8_CONFIG):
        cfg.check_protected(full_names, quant.PROTECTED)
        matched = [n for n in full_names if cfg.scheme_for(n) is not None]
        assert len(matched) == 144
        assert "info_sharing.proj_embed" not in matched
    mxfp8 = [n for n in full_names if (m := quant.FP8_CONFIG.scheme_for(n)) is not None and m[0] == "mxfp8"]
    assert len(mxfp8) == 14
