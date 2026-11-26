"""CPU unit tests for RIPE++ (architecture, subpixel NMS, coordinates, weights loading, specs, and pipeline)."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

import flashim
from flashim.models.ripepp import RIPEPP, weights


def test_forward_shapes_and_describe_keypoints():
    model = RIPEPP(threshold=-1e6, nms_size=3, subpixel_sampling=True).eval()
    image = torch.rand(1, 3, 64, 80)
    with torch.inference_mode():
        kpts, scores, desc = model.detect(image, top_k=32)
    assert kpts.ndim == 2 and kpts.shape[1] == 2 and 0 < kpts.shape[0] <= 32
    assert scores.shape == (kpts.shape[0],)
    assert desc.shape == (kpts.shape[0], 256)
    assert torch.allclose(desc.norm(dim=-1), torch.ones(len(kpts)), atol=1e-5)
    with torch.inference_mode():
        re_desc = model.describe_keypoints(image, kpts)
    assert torch.allclose(re_desc, desc, atol=1e-5)

    # Empty detection (threshold above heatmap max) returns empty tensors cleanly
    model_empty = RIPEPP(threshold=1e6, nms_size=3, subpixel_sampling=True).eval()
    with torch.inference_mode():
        ek, es, ed = model_empty.detect(image, top_k=32)
        ed2 = model_empty.describe_keypoints(image, ek)
    assert ek.shape == (0, 2) and es.shape == (0,) and ed.shape == (0, 256) and ed2.shape == (0, 256)
    assert model.matcher_keypoint_offset == 0.5 and model.default_top_k == 2048


def test_subpixel_offset_matches_softmax_weighted_neighbors():
    model = RIPEPP(threshold=0.5, nms_size=3, subpixel_sampling=True, sub_pixel_temp=0.5).eval()
    heat = torch.zeros(1, 1, 16, 16)
    # Peak at (x=7, y=5) with right neighbor stronger than left -> positive dx, zero dy
    heat[0, 0, 5, 7] = 2.0
    heat[0, 0, 5, 8] = 1.5
    heat[0, 0, 5, 6] = 0.0
    feats = [torch.randn(1, c, 16 // s, 16 // s) for c, s in ((64, 1), (128, 2), (256, 4), (512, 8))]
    model.heatmap = lambda _img: (heat, feats)
    with torch.inference_mode():
        kpts, scores, _ = model.detect(torch.zeros(1, 3, 16, 16), top_k=10)
    assert kpts.shape == (1, 2)
    assert kpts[0, 0].item() > 7.0 and kpts[0, 0].item() < 7.5
    assert abs(kpts[0, 1].item() - 5.0) < 1e-6
    assert abs(scores[0].item() - 1.0) < 1e-6


def test_batchnorm_fold_is_exact():
    model = RIPEPP().eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.uniform_(m.weight, 0.5, 1.5)
            torch.nn.init.normal_(m.bias, 0.0, 0.1)
            torch.nn.init.normal_(m.running_mean, 0.0, 0.1)
            torch.nn.init.uniform_(m.running_var, 0.5, 1.5)
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        h_before, _ = model.heatmap(image)
    expected = len(model.batchnorm_pairs())
    folded = model.fold_batchnorm()
    assert len(folded) == expected > 0
    with torch.inference_mode():
        h_after, _ = model.heatmap(image)
    assert torch.allclose(h_before, h_after, atol=1e-4, rtol=1e-4)


def test_weights_variant_aliases_and_wrapped_ckpt(tmp_path):
    assert weights.resolve_variant("default") == "default"
    assert weights.resolve_variant("aachen") == "aachen"
    assert weights.resolve_variant("aachen_day_night") == "aachen"
    assert weights.resolve_variant("scared") == "scared"
    with pytest.raises(ValueError, match="Unknown RIPE\\+\\+ variant"):
        weights.resolve_variant("nope")

    sd = OrderedDict({"a": torch.tensor([1.0, 2.0])})
    ckpt_path = tmp_path / "wrapped.ckpt"
    torch.save({"state_dict": sd, "config": {"descriptor_dim": 256}, "step": 100}, ckpt_path)
    loaded = weights.load_state_dict(path=ckpt_path)
    assert list(loaded.keys()) == ["a"] and torch.equal(loaded["a"], sd["a"])


def test_weights_lightglue_extraction(tmp_path):
    w = torch.randn(3 * 4 * 8, 32)
    lg_path = tmp_path / "lg.tar"
    torch.save(
        {
            "model": OrderedDict(
                {
                    "extractor.net.weight": torch.zeros(1),
                    "matcher.confidence_thresholds": torch.zeros(9),
                    "matcher.transformers.0.self_attn.Wqkv.weight": w,
                    "matcher.posenc.Wr.weight": torch.ones(4, 2),
                }
            )
        },
        lg_path,
    )
    loaded = weights.load_lightglue_state_dict(num_heads=4, num_layers=1, path=lg_path)
    assert "extractor.net.weight" not in loaded
    assert "confidence_thresholds" not in loaded
    assert set(loaded.keys()) == {"transformers.0.self_attn.Wqkv.weight", "posenc.Wr.weight"}
    assert loaded["transformers.0.self_attn.Wqkv.weight"].shape == w.shape


def test_specs_registered_and_resolved():
    available = set(flashim.available_models())
    for name in ("ripepp", "ripepp-aachen", "ripepp-scared", "ripepp-lightglue"):
        assert name in available

    assert flashim.resolve("ripepp", "bf16").preset == "bf16"
    assert flashim.resolve("ripepp", "int8").preset == "int8"
    assert flashim.resolve("ripepp", "fp8").preset == "fp8"
    # Variants without calibration datasets resolve int8/fp8 -> bf16 via build.NEAREST
    for variant_model in ("ripepp-aachen", "ripepp-scared"):
        assert flashim.resolve(variant_model, "bf16").preset == "bf16"
        assert flashim.resolve(variant_model, "fp8").preset == "bf16"
        assert flashim.resolve(variant_model, "int8").preset == "bf16"

    lg_cfg = flashim.resolve("ripepp-lightglue", "bf16")
    assert lg_cfg.components["extractor"].model == "ripepp"
    assert lg_cfg.components["matcher"].model == "lightglue"
    assert lg_cfg.components["matcher"].variant == "ripepp"
    # MXFP8 is omitted on ripepp-lightglue (resolves fp8 -> int8 via build.NEAREST)
    assert flashim.resolve("ripepp-lightglue", "fp8").preset == "int8"


def test_fraunhofer_tls_hint_on_download_failure(monkeypatch):
    def fake_find_weights(ckpt, *, path=None, env=None):
        raise FileNotFoundError(f"downloading {ckpt.url} failed (CERTIFICATE_VERIFY_FAILED)")

    monkeypatch.setattr(weights, "find_weights", fake_find_weights)
    for variant in ("aachen", "scared"):
        with pytest.raises(FileNotFoundError, match="GEANT TLS ECC 1"):
            weights.load_state_dict(variant=variant)
    # Default variant is hosted on Hugging Face, so it should not append the Fraunhofer TLS hint.
    with pytest.raises(FileNotFoundError) as exc_info:
        weights.load_state_dict(variant="default")
    assert "GEANT TLS ECC 1" not in str(exc_info.value)
