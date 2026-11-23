"""CPU unit tests for MASt3R architecture, CroCo backbone reuse, preprocessing, and spec registration."""

from __future__ import annotations

import torch

import flashim
from flashim.layers import TransformerBlock
from flashim.models.mast3r.config import MASt3RConfig
from flashim.models.mast3r.croco import CroCoStereoBackbone, DPTOutputAdapter, RoPE2D
from flashim.models.mast3r.model import MASt3R, prepare_image
from flashim.models.mast3r.spec import SPEC
from flashim.quant.apply import quantize_model
from flashim.quant.passes.dynamic_transformer_block import DynamicInt8TransformerBlock, fuse_dynamic_blocks


def _tiny_config() -> MASt3RConfig:
    return MASt3RConfig(
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
        local_feat_dim=8,
        hidden_dim_factor=2.0,
        resize=64,
        subsample=4,
        border=1,
    )


def test_registered_in_flashim():
    cfg = flashim.resolve("mast3r", "bf16")
    assert cfg.precision.compute_dtype == "fp16"
    assert SPEC.name == "mast3r"
    assert set(SPEC.presets) >= {"fp32", "bf16"}


def test_tiny_forward_and_match_shapes():
    cfg = _tiny_config()
    model = MASt3R(cfg).eval()
    assert isinstance(model, CroCoStereoBackbone)
    assert all(isinstance(b, TransformerBlock) for b in model.enc_blocks)

    img1 = torch.randn(1, 3, 48, 64)
    img2 = torch.randn(1, 3, 64, 48)  # mixed aspect ratios
    with torch.inference_mode():
        res1, res2 = model(img1, img2)
    assert res1["pts3d"].shape == (1, 48, 64, 3)
    assert res1["conf"].shape == (1, 48, 64)
    assert res1["desc"].shape == (1, 48, 64, cfg.local_feat_dim)
    assert res1["desc_conf"].shape == (1, 48, 64)
    assert res2["pts3d_in_other_view"].shape == (1, 64, 48, 3)
    assert res2["desc"].shape == (1, 64, 48, cfg.local_feat_dim)
    # Unit-norm descriptors
    torch.testing.assert_close(res1["desc"].norm(dim=-1), torch.ones(1, 48, 64), atol=1e-5, rtol=1e-5)

    # End-to-end ImageMatcher.match on synthetic [1, 3, H, W] in [0, 1]
    a = torch.rand(1, 3, 64, 80)
    b = torch.rand(1, 3, 64, 80)
    matches = model.match(a, b, resize=64)
    assert matches.keypoints_A.shape[1] == 2
    assert matches.keypoints_B.shape[1] == 2
    assert len(matches) == matches.confidence.shape[0]


def test_prepare_image_geometry():
    # Landscape 1000x750 -> resized 512x384 -> exact multiple of 16
    img = torch.rand(1, 3, 750, 1000)
    out, (oh, ow, nh, nw, cy, cx) = prepare_image(img, resize=512, patch_size=16)
    assert (oh, ow, nh, nw) == (750, 1000, 384, 512)
    assert (cy, cx) == (0, 0)
    assert out.shape == (1, 3, 384, 512)
    assert out.min() >= -1.0 and out.max() <= 1.0

    # Square image with square_ok=False -> cropped to 4:3 (384x512)
    sq = torch.rand(1, 3, 600, 600)
    out_sq, _ = prepare_image(sq, resize=512, patch_size=16, square_ok=False)
    assert out_sq.shape == (1, 3, 384, 512)


def test_rope2d_and_dpt_odd_patch_grid():
    rope = RoPE2D(freq=100.0)
    sin, cos = rope.for_grid(16, 3, 5, torch.device("cpu"), torch.float32)
    assert sin.shape == (15, 16) and cos.shape == (15, 16)
    # Odd patch count (3x5 -> 48x80 pixels) tests the DPT refinenet4 crop branch
    dpt = DPTOutputAdapter(
        num_channels=4, patch_size=16, hooks=(0, 1, 2, 3),
        dim_tokens=(32, 16, 16, 16), layer_dims=(8, 16, 32, 64), feature_dim=16, last_dim=8,
    )
    tokens = [torch.randn(1, 15, c) for c in (32, 16, 16, 16)]
    out = dpt(tokens, (48, 80))
    assert out.shape == (1, 4, 48, 80)


def test_dynamic_int8_quant_and_block_fusion():
    from flashim.models.mast3r import quant

    cfg = _tiny_config()
    model = MASt3R(cfg).eval()
    report = quantize_model(model, quant.INT8_CONFIG, None, protected=quant.PROTECTED)
    assert len(report.linears) > 0
    fused = fuse_dynamic_blocks(model)
    assert len(fused) == cfg.enc_depth
    assert all(isinstance(b, DynamicInt8TransformerBlock) for b in model.enc_blocks)
    fused_dec = quant.fuse_dynamic_decoder_blocks(model)
    assert len(fused_dec) == 2 * cfg.dec_depth
    assert all(isinstance(b, quant.DynamicInt8DecoderBlock) for b in (*model.dec_blocks, *model.dec_blocks2))
    x1, x2 = torch.randn(1, 3, 48, 64), torch.randn(1, 3, 48, 64)
    with torch.inference_mode():
        out1, out2 = model(x1, x2)
        d1, c1, d2, c2 = model._forward_descriptors_eager(x1, x2)
    assert torch.isfinite(out1["desc"]).all() and torch.isfinite(out2["pts3d_in_other_view"]).all()
    assert torch.equal(out1["desc"], d1) and torch.equal(out1["desc_conf"], c1)
    assert torch.equal(out2["desc"], d2) and torch.equal(out2["desc_conf"], c2)


def test_layernorm_orthogonal_commutes_with_layernorm():
    from flashim.quant.rotation import layernorm_orthogonal

    torch.manual_seed(0)
    for d, block_size in ((64, None), (768, None), (1024, None), (1024, 16)):
        q = layernorm_orthogonal(d, block_size=block_size, dtype=torch.float64)
        eye = torch.eye(d, dtype=torch.float64)
        torch.testing.assert_close(q.T @ q, eye, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(q @ torch.ones(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64), atol=1e-12, rtol=0.0)
        x = torch.randn(3, d, dtype=torch.float64) * 5.0 + 2.0
        ln_x = torch.nn.functional.layer_norm(x, (d,))
        ln_xq = torch.nn.functional.layer_norm(x @ q, (d,))
        torch.testing.assert_close(ln_xq, ln_x @ q, atol=1e-12, rtol=1e-12)


def test_rotate_mast3r_exact_fp32():
    from flashim.models.mast3r import quant

    torch.manual_seed(0)
    cfg = _tiny_config()
    ref_model = MASt3R(cfg).eval()
    rot_model = MASt3R(cfg).eval()
    rot_model.load_state_dict(ref_model.state_dict(), strict=True)
    assert quant.rotate_mast3r(rot_model) > 0

    x1, x2 = torch.randn(1, 3, 48, 64), torch.randn(1, 3, 64, 48)
    with torch.inference_mode():
        r1, r2 = ref_model(x1, x2)
        o1, o2 = rot_model(x1, x2)
    for k in ("pts3d", "conf", "desc", "desc_conf"):
        torch.testing.assert_close(o1[k], r1[k], atol=1e-4, rtol=1e-4)
    for k in ("pts3d_in_other_view", "conf", "desc", "desc_conf"):
        torch.testing.assert_close(o2[k], r2[k], atol=1e-4, rtol=1e-4)


def test_dynamic_int8_block_rotate_cpu():
    from flashim.models.mast3r import quant
    from flashim.quant.methods.int8_linear import DynamicInt8Linear
    from flashim.quant.rotation import hadamard
    from flashim.utils import set_submodule

    torch.manual_seed(0)
    cfg = _tiny_config()
    model = MASt3R(cfg).eval()
    quant.rotate_mast3r(model)
    r16 = hadamard(16)
    for name, mod in list(model.named_modules()):
        if isinstance(mod, torch.nn.Linear) and name.endswith(".mlp.fc2"):
            set_submodule(model, name, DynamicInt8Linear.from_linear(mod, block_rotate=r16))
        elif isinstance(mod, torch.nn.Linear) and any(
            name.startswith(p) for p in ("enc_blocks.", "dec_blocks.", "dec_blocks2.", "decoder_embed")
        ):
            set_submodule(model, name, DynamicInt8Linear.from_linear(mod))
    assert len(fuse_dynamic_blocks(model)) == cfg.enc_depth
    assert len(quant.fuse_dynamic_decoder_blocks(model)) == 2 * cfg.dec_depth
    x1, x2 = torch.randn(1, 3, 48, 64), torch.randn(1, 3, 48, 64)
    with torch.inference_mode():
        out1, out2 = model(x1, x2)
    assert torch.isfinite(out1["desc"]).all() and torch.isfinite(out2["pts3d_in_other_view"]).all()


def test_dynamic_vit_block_hybrid_mxfp8_fc2_cpu():
    from flashim.quant.methods.int8_linear import DynamicInt8Linear
    from flashim.quant.methods.linear import StaticMXFP8Linear
    from flashim.quant.passes.dynamic_transformer_block import DynamicInt8TransformerBlock, is_fusable

    torch.manual_seed(0)
    block = TransformerBlock(64, 4, eps=1e-6, layer_scale=False).to(torch.bfloat16).eval()
    block.attn.qkv = DynamicInt8Linear.from_linear(block.attn.qkv)
    block.attn.proj = DynamicInt8Linear.from_linear(block.attn.proj)
    block.mlp.fc1 = DynamicInt8Linear.from_linear(block.mlp.fc1)
    block.mlp.fc2 = StaticMXFP8Linear.from_linear(block.mlp.fc2, torch.zeros(block.mlp.fc2.in_features // 32, dtype=torch.int8))
    assert is_fusable(block)
    fused = DynamicInt8TransformerBlock(block, torch.device("cpu"))
    x = torch.randn(1, 12, 64, dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(fused(x), block(x), atol=5e-2, rtol=5e-2)
