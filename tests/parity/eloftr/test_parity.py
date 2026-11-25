"""FlashIM's EfficientLoFTR against the vendored upstream implementation (zju3dv/EfficientLoFTR @ ffd4a46440 in
``tests/oracles/eloftr``; same checkpoint loaded from FlashIM's local copy).

Upstream's ``FineMatching.forward`` wraps its fine-correlation matmul in ``torch.autocast(enabled=True, device_type='cuda')``
unless ``validate=True`` (even in FP32 mode), which injects FP16 rounding noise into ~25% of pixel argmaxes inside flat
8x8 patches. The FP32 parity test sets ``oracle.fine_matching.validate = True`` to compare true FP32 numerics (just as
``tests/oracles/loma/loma_oracle.py`` disables hard-coded AMP for its FP32 reference).
"""

from __future__ import annotations

import copy
import sys
import types

import pytest
import torch
import torch.nn.functional as F

import flashim
from flashim.models import sparse
from flashim.models.eloftr import weights
from flashim.models.romav2.model import load_image

from ...conftest import ASSETS, ORACLES

pytestmark = [pytest.mark.oracle, pytest.mark.weights, pytest.mark.cuda, pytest.mark.slow]

ELOFTR_ROOT = ORACLES / "eloftr"
PAIR = (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg")


def _import_upstream():
    """Import ``src.loftr`` from ``tests/oracles/eloftr`` without colliding with other oracles' ``src`` packages."""
    kornia = pytest.importorskip("kornia")
    pytest.importorskip("yacs")
    pytest.importorskip("einops")
    if not (ELOFTR_ROOT / "src" / "loftr" / "loftr.py").exists():
        pytest.skip(f"needs the vendored ELoFTR oracle at {ELOFTR_ROOT}")

    # Stub unused upstream training/logging imports and kornia 0.8's moved create_meshgrid.
    stubs: dict[str, types.ModuleType] = {}
    for name in ("loguru", "joblib", "pytorch_lightning", "pytorch_lightning.utilities"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            stubs[name] = mod
            sys.modules[name] = mod
    if "loguru" in stubs:
        stubs["loguru"].logger = types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None, log=lambda *a, **k: None
        )
        stubs["loguru"]._Logger = object
    elif not hasattr(sys.modules["loguru"], "_Logger"):
        sys.modules["loguru"]._Logger = object
    if "pytorch_lightning.utilities" in stubs:
        stubs["pytorch_lightning.utilities"].rank_zero_only = types.SimpleNamespace(rank=0)
    if "kornia.utils.grid" not in sys.modules:
        grid_mod = types.ModuleType("kornia.utils.grid")
        grid_mod.create_meshgrid = kornia.geometry.create_meshgrid
        stubs["kornia.utils.grid"] = grid_mod
        sys.modules["kornia.utils.grid"] = grid_mod

    saved_src = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
    sys.path.insert(0, str(ELOFTR_ROOT))
    try:
        from src.loftr import LoFTR as UpstreamELoFTR
        from src.loftr import full_default_cfg, opt_default_cfg, reparameter
    finally:
        sys.path.remove(str(ELOFTR_ROOT))
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules.pop(k, None)
        sys.modules.update(saved_src)
        for name in stubs:
            sys.modules.pop(name, None)
    return UpstreamELoFTR, full_default_cfg, opt_default_cfg, reparameter


def _prepared_pair(size0: tuple[int, int], size1: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g0 = sparse.rgb_to_gray(F.interpolate(load_image(str(PAIR[0]), device), size=size0, mode="bilinear", antialias=True))
    g1 = sparse.rgb_to_gray(F.interpolate(load_image(str(PAIR[1]), device), size=size1, mode="bilinear", antialias=True))
    return g0, g1


@pytest.mark.parametrize("variant", ["full", "opt"])
@pytest.mark.parametrize("shapes", [((480, 640), (480, 640)), ((480, 640), (512, 640))], ids=["same", "diff"])
def test_fp32_matches_oracle(variant, shapes, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    UpstreamELoFTR, full_default_cfg, opt_default_cfg, reparameter = _import_upstream()
    try:
        state = weights.load_state_dict()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    cfg = copy.deepcopy(full_default_cfg if variant == "full" else opt_default_cfg)
    upstream = UpstreamELoFTR(config=cfg)
    upstream.load_state_dict(state)
    upstream = reparameter(upstream).cuda().eval()
    upstream.fine_matching.validate = True  # true FP32 reference (disable upstream's hard-coded FP16 fine autocast)

    ours = flashim.load("eloftr", "fp32", device="cuda", overrides={"variant": variant})
    g0, g1 = _prepared_pair(*shapes, torch.device("cuda"))
    batch = {"image0": g0.clone(), "image1": g1.clone()}
    with torch.inference_mode():
        upstream(batch)
        k0, k1, conf = ours.match_prepared(g0, g1)

    uk0, uk1, uconf = batch["mkpts0_f"], batch["mkpts1_f"], batch["mconf"]
    assert len(conf) > 50
    assert len(conf) == len(uconf)
    assert (k0 - uk0).abs().max().item() < 1e-4
    assert (k1 - uk1).abs().max().item() < 1e-3
    assert (conf - uconf).abs().max().item() < 1e-4


@pytest.mark.parametrize("preset", ["bf16", "int8", "fp8"])
def test_reduced_precision_tracks_fp32(preset):
    fp32 = flashim.load("eloftr", "fp32", device="cuda")
    model = flashim.load("eloftr", preset, device="cuda")
    if preset in ("int8", "fp8") and not model.build_report.fallbacks:
        assert model.build_report.quant["linears"] == (25 if preset == "int8" else 24)
    g0, g1 = _prepared_pair((480, 640), (480, 640), torch.device("cuda"))
    with torch.inference_mode():
        ref0, ref1, ref_c = fp32.match_prepared(g0, g1)
        k0, k1, conf = model.match_prepared(g0, g1)
    assert abs(len(conf) - len(ref_c)) <= 0.10 * len(ref_c) + 5
    d0 = torch.cdist(ref0.float(), k0.float(), compute_mode="donot_use_mm_for_euclid_dist")
    d1 = torch.cdist(ref1.float(), k1.float(), compute_mode="donot_use_mm_for_euclid_dist")
    # Stage-1 fine matching picks any pixel inside each 8x8 coarse cell on both images, so compare within 1 cell (8 px).
    agree = ((d0 < 8.0) & (d1 < 8.0)).any(dim=1).float().mean().item()
    min_agree = 0.90 if preset == "bf16" else 0.85
    assert agree >= min_agree, f"{preset} match agreement vs fp32 dropped to {agree:.4f}"


def test_match_image_interface():
    model = flashim.load("eloftr", "bf16", device="cuda")
    res = model.match(PAIR[0], PAIR[1], resize=640, max_matches=200)
    assert len(res.confidence) > 50
    assert len(res.confidence) <= 200
    assert res.keypoints_A.shape == (len(res.confidence), 2)
    assert res.keypoints_B.shape == (len(res.confidence), 2)
    assert torch.isfinite(res.keypoints_A).all() and torch.isfinite(res.keypoints_B).all()
