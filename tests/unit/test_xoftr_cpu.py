"""CPU unit tests for the native XoFTR implementation (no weights or GPU required)."""

from __future__ import annotations

import sys

import pytest
import torch

import flashim
from flashim.models.xoftr import XoFTR, XoFTRConfig
from flashim.paths import ORACLES


def _tiny_pair(h0: int = 64, w0: int = 64, h1: int = 64, w1: int = 64, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    g0 = torch.rand(1, 1, h0, w0, generator=gen)
    g1 = torch.rand(1, 1, h1, w1, generator=gen)
    return g0, g1


def test_state_dict_matches_upstream_keys():
    sys.path.insert(0, str(ORACLES / "xoftr"))
    try:
        from config.default import get_cfg_defaults
        from xoftr import XoFTR as UpstreamXoFTR
    finally:
        sys.path.remove(str(ORACLES / "xoftr"))

    def lower(cfg):
        return {k.lower(): lower(v) for k, v in cfg.items()} if hasattr(cfg, "items") else cfg

    cfg = lower(get_cfg_defaults(inference=True))["xoftr"]
    ref = UpstreamXoFTR(cfg)
    ours = XoFTR()
    assert set(ours.state_dict().keys()) == set(ref.state_dict().keys())
    for k, v in ref.state_dict().items():
        assert ours.state_dict()[k].shape == v.shape, k


@pytest.mark.parametrize("shapes", [(64, 64, 64, 64), (64, 80, 72, 64)])
def test_forward_shapes_and_empty_matches(shapes):
    model = XoFTR(coarse_threshold=0.0, fine_threshold=0.0).eval()
    g0, g1 = _tiny_pair(*shapes)
    k0, k1, conf = model.match_prepared(g0, g1)
    assert k0.ndim == 2 and k0.shape[1] == 2
    assert k1.shape == k0.shape
    assert conf.shape == (k0.shape[0],)
    assert torch.isfinite(k0).all() and torch.isfinite(k1).all() and torch.isfinite(conf).all()

    # Very high coarse threshold -> 0 matches cleanly.
    model.threshold = 1.1
    ek0, ek1, econf = model.match_prepared(g0, g1)
    assert ek0.shape == (0, 2) and ek1.shape == (0, 2) and econf.shape == (0,)


def test_denser_mode_runs():
    model = XoFTR(coarse_threshold=0.0, fine_threshold=0.0, denser=True).eval()
    g0, g1 = _tiny_pair(64, 64, 64, 64)
    k0, k1, conf = model.match_prepared(g0, g1)
    assert k0.shape == k1.shape and conf.shape == (len(k0),)


def test_fold_batchnorm_is_exact_on_cpu():
    torch.manual_seed(7)
    model = XoFTR(coarse_threshold=0.0, fine_threshold=0.0).eval()
    # Give every BN non-trivial stats so folding is exercised.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
            m.running_mean.uniform_(-0.3, 0.3)
            m.running_var.uniform_(0.5, 2.0)
    g0, g1 = _tiny_pair(64, 64, 64, 64, seed=3)
    k0_ref, k1_ref, conf_ref = model.match_prepared(g0, g1)

    folded = model.fold_batchnorm()
    assert len(folded) == 16  # 1 stem + 6 blocks * 2 + 2 downsamples + 1 fine conv_merge depthwise
    assert "fine_process.conv_merge.1" in folded

    k0, k1, conf = model.match_prepared(g0, g1)
    assert k0.shape == k0_ref.shape
    assert (k0 - k0_ref).abs().max().item() < 1e-4
    assert (k1 - k1_ref).abs().max().item() < 1e-4
    assert (conf - conf_ref).abs().max().item() < 1e-5


def test_match_interface_scaling_and_max_matches():
    model = XoFTR(XoFTRConfig(coarse_threshold=0.0, fine_threshold=0.0, default_resize=64)).eval()
    img_a = torch.rand(1, 3, 96, 128)
    img_b = torch.rand(1, 3, 80, 96)
    res = model.match(img_a, img_b, resize=64, max_matches=5)
    assert res.keypoints_A.shape[1] == 2 and res.keypoints_B.shape[1] == 2
    assert len(res.confidence) <= 5
    assert res.keypoints_A.shape == res.keypoints_B.shape


def test_registry_and_resolve():
    assert "xoftr" in flashim.available_models()
    for preset in ("fp32", "bf16", "int8", "fp8"):
        cfg = flashim.resolve("xoftr", preset)
        assert cfg.model == "xoftr"
        assert cfg.precision.compute_dtype == ("fp32" if preset == "fp32" else "bf16")
