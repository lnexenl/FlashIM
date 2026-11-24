"""CPU unit tests for DUSt3R architecture, CroCo backbone reuse, rotation, quantization, and spec registration."""

from __future__ import annotations

import torch

import flashim
from flashim.layers import TransformerBlock
from flashim.models.dust3r import quant
from flashim.models.dust3r.config import DUSt3RConfig
from flashim.models.dust3r.model import DUSt3R
from flashim.models.dust3r.quant import DynamicInt8DecoderBlock
from flashim.models.dust3r.spec import SPEC
from flashim.models.mast3r.croco import CroCoStereoBackbone
from flashim.quant.apply import quantize_model
from flashim.quant.passes.dynamic_transformer_block import DynamicInt8TransformerBlock, fuse_dynamic_blocks


def _tiny_config() -> DUSt3RConfig:
    return DUSt3RConfig(
        img_size=(64, 64),
        patch_size=16,
        enc_embed_dim=64,
        enc_depth=2,
        enc_num_heads=4,
        dec_embed_dim=32,
        dec_depth=4,
        dec_num_heads=4,
        mlp_ratio=2,
        dpt_feature_dim=32,
        resize=64,
    )


def test_registered_in_flashim():
    cfg = flashim.resolve("dust3r", "bf16")
    assert cfg.precision.compute_dtype == "fp16"
    assert SPEC.name == "dust3r"
    assert set(SPEC.presets) >= {"fp32", "bf16"}


def test_tiny_forward_and_match_shapes():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = DUSt3R(cfg).eval()
    assert isinstance(model, CroCoStereoBackbone)
    assert all(isinstance(b, TransformerBlock) for b in model.enc_blocks)

    img1 = torch.randn(1, 3, 48, 64)
    img2 = torch.randn(1, 3, 64, 48)  # mixed aspect ratios
    with torch.inference_mode():
        res1, res2 = model(img1, img2)
    assert set(res1) == {"pts3d", "conf"}
    assert set(res2) == {"pts3d_in_other_view", "conf"}
    assert res1["pts3d"].shape == (1, 48, 64, 3)
    assert res1["conf"].shape == (1, 48, 64)
    assert res2["pts3d_in_other_view"].shape == (1, 64, 48, 3)
    assert res2["conf"].shape == (1, 64, 48)
    assert (res1["conf"] >= 1.0).all() and (res2["conf"] >= 1.0).all()

    # End-to-end ImageMatcher.match on synthetic [1, 3, H, W] in [0, 1]
    a = torch.rand(1, 3, 64, 80)
    b = torch.rand(1, 3, 64, 80)
    matches = model.match(a, b, resize=64, max_matches=50)
    assert matches.keypoints_A.shape[1] == 2
    assert matches.keypoints_B.shape[1] == 2
    assert len(matches) == matches.confidence.shape[0] <= 50

    # High threshold yields empty Matches cleanly
    empty = model.match(a, b, resize=64, threshold=1e9)
    assert len(empty) == 0 and empty.keypoints_A.shape == (0, 2)


def test_rotate_dust3r_exact_fp32():
    torch.manual_seed(0)
    cfg = _tiny_config()
    ref_model = DUSt3R(cfg).eval()
    rot_model = DUSt3R(cfg).eval()
    rot_model.load_state_dict(ref_model.state_dict(), strict=True)
    assert quant.rotate_dust3r(rot_model) > 0

    x1, x2 = torch.randn(1, 3, 48, 64), torch.randn(1, 3, 64, 48)
    with torch.inference_mode():
        r1, r2 = ref_model(x1, x2)
        o1, o2 = rot_model(x1, x2)
    for k in ("pts3d", "conf"):
        torch.testing.assert_close(o1[k], r1[k], atol=1e-4, rtol=1e-4)
    for k in ("pts3d_in_other_view", "conf"):
        torch.testing.assert_close(o2[k], r2[k], atol=1e-4, rtol=1e-4)


def test_dynamic_int8_quant_and_block_fusion():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = DUSt3R(cfg).eval()
    quant.rotate_dust3r(model)
    quant.attach_fc2_hadamard(model, include_encoder=True, block_size=16)
    report = quantize_model(model, quant.INT8_CONFIG, None, protected=quant.PROTECTED)
    # Every linear in DUSt3R is quantized (no unquantized head MLP).
    assert len(report.linears) > 0
    assert not any(isinstance(m, torch.nn.Linear) for m in model.modules())

    fused_enc = fuse_dynamic_blocks(model)
    assert len(fused_enc) == cfg.enc_depth
    assert all(isinstance(b, DynamicInt8TransformerBlock) for b in model.enc_blocks)

    fused_dec = quant.fuse_dynamic_decoder_blocks(model)
    assert len(fused_dec) == 2 * cfg.dec_depth
    assert all(isinstance(b, DynamicInt8DecoderBlock) for b in (*model.dec_blocks, *model.dec_blocks2))

    x1, x2 = torch.randn(1, 3, 48, 64), torch.randn(1, 3, 48, 64)
    with torch.inference_mode():
        out1, out2 = model(x1, x2)
    assert torch.isfinite(out1["pts3d"]).all() and torch.isfinite(out1["conf"]).all()
    assert torch.isfinite(out2["pts3d_in_other_view"]).all() and torch.isfinite(out2["conf"]).all()


def test_hybrid_fp8_quant_and_block_fusion_cpu():
    from flashim.quant.calibration import CalibrationTable, ModuleScales

    torch.manual_seed(0)
    cfg = _tiny_config()
    model = DUSt3R(cfg).eval()
    model.set_compute_dtype(torch.bfloat16)
    quant.rotate_dust3r(model)
    quant.attach_fc2_hadamard(model, include_encoder=False, block_size=16)
    enc_fc2 = {
        f"enc_blocks.{i}.mlp.fc2": ModuleScales(
            input_exponent=torch.zeros(cfg.enc_embed_dim * cfg.mlp_ratio // 32, dtype=torch.int8)
        )
        for i in range(cfg.enc_depth)
    }
    report = quantize_model(model, quant.FP8_CONFIG, CalibrationTable(linear=enc_fc2), protected=quant.PROTECTED)
    assert len(report.linears) > 0
    assert not any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert len(fuse_dynamic_blocks(model)) == cfg.enc_depth
    assert len(quant.fuse_dynamic_decoder_blocks(model)) == 2 * cfg.dec_depth
    x1, x2 = torch.randn(1, 3, 48, 64, dtype=torch.bfloat16), torch.randn(1, 3, 48, 64, dtype=torch.bfloat16)
    with torch.inference_mode():
        out1, out2 = model(x1, x2)
    assert torch.isfinite(out1["pts3d"]).all() and torch.isfinite(out2["pts3d_in_other_view"]).all()
