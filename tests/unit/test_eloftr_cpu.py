"""CPU unit tests for EfficientLoFTR: RepVGG reparameterization, window gather, config / spec resolution, synthetic
forward passes for ``full`` and ``opt``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import flashim
from flashim import ops
from flashim.models.eloftr.config import ELoFTRConfig
from flashim.models.eloftr.model import AggregatedEncoderLayer, ELoFTR, RepVGGBlock, rope_2d_tables


def test_repvgg_reparameterization_matches_multibranch():
    torch.manual_seed(0)
    for keep_identity in (False, True):
        for in_ch, out_ch, stride in ((1, 16, 2), (16, 16, 1), (16, 32, 2)):
            block = RepVGGBlock(in_ch, out_ch, 3, stride=stride, padding=1)
            for m in block.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.running_mean.uniform_(-1, 1)
                    m.running_var.uniform_(0.5, 2.0)
                    m.weight.data.uniform_(0.5, 2.0)
                    m.bias.data.uniform_(-0.5, 0.5)
            block.eval()
            x = torch.randn(2, in_ch, 32, 32)
            with torch.inference_mode():
                ref = block(x)
                assert block.switch_to_deploy(keep_identity=keep_identity) is True
                assert block.switch_to_deploy(keep_identity=keep_identity) is False
                assert hasattr(block, "rbr_reparam")
                assert not hasattr(block, "rbr_dense")
                assert not hasattr(block, "rbr_1x1")
                assert hasattr(block, "id_scale") == (keep_identity and in_ch == out_ch and stride == 1)
                out = block(x)
            assert (out - ref).abs().max().item() < 1e-5


def test_coarse_layer_pack_kv_and_fuse_mlp_matches_unfused():
    torch.manual_seed(7)
    for rope_on in (True, False):
        layer = AggregatedEncoderLayer(64, 4, agg_size0=4, agg_size1=4, rope=rope_on).eval()
        x = torch.randn(1, 64, 16, 20)
        src = torch.randn(1, 64, 16, 20)
        rope = rope_2d_tables(64, 4, 5, (832, 832, 832, 832), torch.device("cpu")) if rope_on else None
        with torch.inference_mode():
            ref = layer(x, src, rope)
            assert layer.pack_kv() is True
            assert layer.fuse_mlp() is True
            out = layer(x, src, rope)
        assert (out - ref).abs().max().item() < 1e-4


def test_gather_windows_matches_unfold():
    torch.manual_seed(1)
    model = ELoFTR(ELoFTRConfig()).eval()
    hc0, wc0, hc1, wc1 = 4, 5, 4, 6
    f0 = torch.randn(1, 64, hc0 * 8, wc0 * 8)
    f1 = torch.randn(1, 64, hc1 * 8, wc1 * 8)
    i_ids = torch.tensor([0, 3, 7, 19])
    j_ids = torch.tensor([1, 5, 11, 23])

    fw0, fw1 = model._gather_windows(f0, f1, i_ids, j_ids, hc0, wc0, hc1, wc1)

    u0 = F.unfold(f0, kernel_size=(8, 8), stride=8, padding=0).reshape(1, 64, 64, -1).permute(0, 3, 2, 1)[0, i_ids]
    u1 = F.unfold(f1, kernel_size=(10, 10), stride=8, padding=1).reshape(1, 64, 100, -1).permute(0, 3, 2, 1)[0, j_ids]
    assert torch.equal(fw0, u0)
    assert torch.equal(fw1, u1)

    # Also verify upsample_gather_windows reference against full 2x upsample + _gather_windows.
    f0_half = torch.randn(1, 64, hc0 * 4, wc0 * 4)
    f1_half = torch.randn(1, 64, hc1 * 4, wc1 * 4)
    ref0, ref1 = model._gather_windows(
        F.interpolate(f0_half, scale_factor=2.0, mode="bilinear", align_corners=False),
        F.interpolate(f1_half, scale_factor=2.0, mode="bilinear", align_corners=False),
        i_ids,
        j_ids,
        hc0,
        wc0,
        hc1,
        wc1,
    )
    got0 = ops.eloftr_upsample_gather_windows(f0_half, i_ids, hc0, wc0, 8, 0)
    got1 = ops.eloftr_upsample_gather_windows(f1_half, j_ids, hc1, wc1, 10, 1)
    assert torch.equal(got0, ref0)
    assert torch.equal(got1, ref1)


def test_rope_2d_tables_shape_and_dtype():
    cos, sin = rope_2d_tables(256, 3, 4, (832, 832, 832, 832), torch.device("cpu"))
    assert cos.shape == (1, 1, 12, 256)
    assert sin.shape == (1, 1, 12, 256)
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16


def test_config_and_spec_resolution():
    cfg_default = flashim.resolve("eloftr")
    assert cfg_default.preset == "bf16"
    assert "reparameterize" in cfg_default.optimize.passes
    assert "fold_batchnorm" in cfg_default.optimize.passes

    cfg_opt = flashim.resolve("eloftr", "fp32", overrides={"variant": "opt", "arch.npe": [832, 832, 1152, 1152]})
    assert cfg_opt.variant == "opt"
    assert cfg_opt.arch["npe"] == [832, 832, 1152, 1152]

    with pytest.raises(ValueError, match="unknown variant"):
        flashim.resolve("eloftr", overrides={"variant": "bogus"})


def test_synthetic_forward_full_and_opt():
    torch.manual_seed(2)
    x0 = torch.rand(1, 1, 64, 64)
    x1 = torch.rand(1, 1, 64, 96)
    for variant, thr in (("full", 0.0), ("opt", -1e9)):
        model = ELoFTR(ELoFTRConfig(variant=variant, threshold=thr, border_rm=0)).eval()
        assert len(model.reparameterize()) == 21
        assert len(model.fold_batchnorm()) == 2
        k0, k1, conf = model.match_prepared(x0, x1)
        assert k0.ndim == 2 and k0.shape[1] == 2
        assert k1.shape == k0.shape
        assert conf.shape == (len(k0),)
        assert torch.isfinite(k0).all() and torch.isfinite(k1).all() and torch.isfinite(conf).all()
