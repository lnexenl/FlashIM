"""CPU unit tests for MINIMA variants (`minima-loftr`, `minima-roma`): config resolution, calibration routing,
LoFTR ``temp_bug_fix`` default, and weight registry (no checkpoints or GPU needed)."""

from __future__ import annotations

import pytest

import flashim
from flashim import build
from flashim.models.loftr import weights as loftr_weights
from flashim.models.loftr.model import LoFTR
from flashim.models.roma import weights as roma_weights
from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable


def test_resolve_minima_loftr_and_variants():
    assert flashim.resolve("loftr", "fp32").variant is None
    assert flashim.resolve("loftr", "bf16", overrides={"variant": "minima"}).variant == "minima"
    for preset in ("fp32", "bf16", "int8", "fp8"):
        cfg = flashim.resolve("minima-loftr", preset)
        assert cfg.model == "minima-loftr"
        assert cfg.variant == "minima"
        assert cfg.preset == preset


def test_resolve_minima_roma_and_variants():
    assert flashim.resolve("roma", "fp32").variant is None
    assert flashim.resolve("roma", "bf16", overrides={"variant": "minima"}).variant == "minima"
    for preset in ("fp32", "bf16", "int8", "fp8"):
        cfg = flashim.resolve("minima-roma", preset)
        assert cfg.model == "minima-roma"
        assert cfg.variant == "minima"
        assert cfg.preset == preset


def test_minima_calibration_table_routing(monkeypatch):
    loaded = []
    for cls in (CalibrationTable, Int8CalibrationTable):
        monkeypatch.setattr(cls, "load", classmethod(lambda cls, path: loaded.append(path.stem)))

    for model_name, variant, preset in (
        ("loftr", None, "int8"),
        ("loftr", "minima", "int8"),
        ("minima-loftr", None, "int8"),
        ("minima-loftr", None, "fp8"),
        ("roma", None, "fp8"),
        ("roma", "minima", "int8"),
        ("minima-roma", None, "int8"),
        ("minima-roma", None, "fp8"),
    ):
        spec = build.get_spec(model_name)
        overlay = {"variant": variant} if variant is not None else {}
        build._load_calibration(spec, flashim.resolve(model_name, preset, overlay), int8=preset == "int8")

    assert loaded == [
        "loftr_int8",
        "loftr_minima_int8",
        "loftr_minima_int8",
        "loftr_minima_fp8",
        "roma_outdoor_fp8_mega",
        "roma_minima_int8_mega",
        "roma_minima_int8_mega",
        "roma_minima_fp8_mega",
    ]


def test_loftr_temp_bug_fix_default_per_variant(monkeypatch):
    """MINIMA-LoFTR and indoor use ``temp_bug_fix=True``; outdoor keeps ``False``."""
    dummy_state = LoFTR().state_dict()
    monkeypatch.setattr(loftr_weights, "load_state_dict", lambda variant, path=None: dummy_state)

    assert LoFTR.from_pretrained(variant="outdoor", device="cpu").temp_bug_fix is False
    assert LoFTR.from_pretrained(variant="indoor", device="cpu").temp_bug_fix is True
    assert LoFTR.from_pretrained(variant="minima", device="cpu").temp_bug_fix is True
    assert LoFTR.from_pretrained(variant="minima", device="cpu", temp_bug_fix=False).temp_bug_fix is False


def test_unknown_variants_raise():
    with pytest.raises(ValueError, match="Unknown LoFTR variant"):
        loftr_weights.load_state_dict("nonexistent")
    with pytest.raises(ValueError, match="Unknown RoMa variant"):
        roma_weights.load_state_dict("nonexistent")


def test_loftr_fused_qkv_and_passes_cpu():
    """Fused QKV/KV projections and fused passes match the unfused LoFTREncoderLayer on CPU."""
    import torch

    from flashim.models.loftr import spec as loftr_spec
    from flashim.models.loftr.model import LocalFeatureTransformer

    torch.manual_seed(0)
    tr_ref = LocalFeatureTransformer(64, 4, ["self", "cross"]).eval()
    tr_opt = LocalFeatureTransformer(64, 4, ["self", "cross"]).eval()
    tr_opt.load_state_dict(tr_ref.state_dict())

    for layer, name in zip(tr_opt.layers, tr_opt.layer_names):
        layer.use_fused_linear_attention = True
        layer.fused_epilogues = True
        if name == "self":
            layer.qkv_proj = loftr_spec._merge_linears((layer.q_proj, layer.k_proj, layer.v_proj))
            layer.q_proj = layer.k_proj = layer.v_proj = None
        else:
            layer.kv_proj = loftr_spec._merge_linears((layer.k_proj, layer.v_proj))
            layer.k_proj = layer.v_proj = None
    tr_opt.batch_self = True

    with torch.inference_mode():
        # Equal-length pair
        a = torch.randn(1, 16, 64)
        b = torch.randn(1, 16, 64)
        r0, r1 = tr_ref(a, b)
        o0, o1 = tr_opt(a, b)
        torch.testing.assert_close(o0, r0, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(o1, r1, atol=1e-4, rtol=1e-4)

        # Unequal-length pair (exercises forward_self_pair)
        b2 = torch.randn(1, 19, 64)
        r0, r1 = tr_ref(a, b2)
        o0, o1 = tr_opt(a, b2)
        torch.testing.assert_close(o0, r0, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(o1, r1, atol=1e-4, rtol=1e-4)


def test_loftr_compile_regions():
    import torch

    from flashim.models.loftr import spec as loftr_spec
    from flashim.quant.methods.conv2d import StaticInt8Conv2d

    m = LoFTR().eval()
    assert loftr_spec._compile_regions(m) == ("backbone",)
    conv = m.backbone.layer1[0].conv1
    m.backbone.layer1[0].conv1 = StaticInt8Conv2d.from_conv(conv, torch.tensor(0.05))
    assert loftr_spec._compile_regions(m) == ()
