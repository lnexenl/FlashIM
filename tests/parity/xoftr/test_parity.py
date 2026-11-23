"""FlashIM's XoFTR against the vendored upstream ``XoFTR`` (same ``weights_xoftr_640.ckpt`` checkpoint).

Note: thermal-visible pairs are not checked into ``assets/``; parity is verified on visible pairs from
``assets/megadepth_0015`` and ``assets/sacre_coeur{1,2}.jpg`` (both modalities pass through the identical grayscale
pipeline).
"""

from __future__ import annotations

import sys

import pytest
import torch

import flashim
from flashim.models import sparse
from flashim.models.romav2.model import load_image
from flashim.models.xoftr import weights
from flashim.paths import ASSETS, MEGADEPTH_IMAGES, ORACLES

pytestmark = [pytest.mark.cuda, pytest.mark.weights, pytest.mark.oracle, pytest.mark.slow]

PAIRS = [
    pytest.param(
        (MEGADEPTH_IMAGES / "3383162411_67cc9af85e_o.jpg", MEGADEPTH_IMAGES / "299056923_ecd980298c_o.jpg"),
        id="megadepth_0015",
    ),
    pytest.param(
        (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg"),
        id="sacre_coeur",
    ),
    pytest.param(
        (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg"),
        id="toronto",
    ),
]


def _lower_config(cfg):
    return {k.lower(): _lower_config(v) for k, v in cfg.items()} if hasattr(cfg, "items") else cfg


@pytest.fixture(scope="module")
def oracle():
    sys.path.insert(0, str(ORACLES / "xoftr"))
    try:
        from config.default import get_cfg_defaults
        from xoftr import XoFTR as UpstreamXoFTR
    finally:
        sys.path.remove(str(ORACLES / "xoftr"))

    try:
        state = weights.load_state_dict()
    except FileNotFoundError:
        pytest.skip("needs weights_xoftr_640.ckpt")
    cfg = _lower_config(get_cfg_defaults(inference=True))["xoftr"]
    model = UpstreamXoFTR(config=cfg)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


@pytest.fixture(scope="module")
def flashim_fp32():
    try:
        return flashim.load("xoftr", "fp32", device="cuda")
    except FileNotFoundError:
        pytest.skip("needs weights_xoftr_640.ckpt")


def _prepare_gray_pair(path_a, path_b, resize: int = 640):
    grays = []
    for path in (path_a, path_b):
        image = load_image(str(path), torch.device("cuda"))
        h, w = image.shape[-2:]
        nh, nw = sparse.target_size(h, w, resize, 8)
        image = torch.nn.functional.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        grays.append(sparse.rgb_to_gray(image))
    return grays[0], grays[1]


@pytest.mark.parametrize("pair", PAIRS)
def test_fp32_matches_oracle(oracle, flashim_fp32, pair, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    if not all(p.exists() for p in pair):
        pytest.skip("needs asset images")

    gray0, gray1 = _prepare_gray_pair(*pair, resize=640)
    with torch.inference_mode():
        batch = {"image0": gray0, "image1": gray1}
        oracle(batch)
        ref_k0, ref_k1, ref_conf = batch["mkpts0_f"], batch["mkpts1_f"], batch["mconf_f"]
        k0, k1, conf = flashim_fp32.match_prepared(gray0, gray1)

    assert len(ref_conf) > 100
    assert abs(len(conf) - len(ref_conf)) <= 0.01 * len(ref_conf) + 2

    # Compare matches as coordinate sets (borderline coarse/fine threshold points can shift indices).
    da = torch.cdist(ref_k0.float(), k0.float(), compute_mode="donot_use_mm_for_euclid_dist")
    db = torch.cdist(ref_k1.float(), k1.float(), compute_mode="donot_use_mm_for_euclid_dist")
    joint = torch.maximum(da, db)
    best_dist, best_idx = joint.min(dim=1)
    matched = best_dist < 0.05
    agree = float(matched.float().mean())
    max_coord_diff = float(best_dist[matched].max()) if matched.any() else float("inf")
    max_conf_diff = float((ref_conf[matched] - conf[best_idx[matched]]).abs().max()) if matched.any() else float("inf")

    assert agree >= 0.99, f"agreement {agree:.4f}, len ours={len(conf)} ref={len(ref_conf)}"
    assert max_coord_diff < 1e-2, f"max coord diff {max_coord_diff}"
    assert max_conf_diff < 1e-3, f"max conf diff {max_conf_diff}"


@pytest.mark.parametrize(("preset", "min_agree"), [("bf16", 0.90), ("int8", 0.85), ("fp8", 0.85)])
def test_reduced_precision_tracks_fp32(flashim_fp32, preset, min_agree):
    pair = (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg")
    if not all(p.exists() for p in pair):
        pytest.skip("needs asset images")
    gray0, gray1 = _prepare_gray_pair(*pair, resize=640)
    with torch.inference_mode():
        rk0, rk1, rc = flashim_fp32.match_prepared(gray0, gray1)
    model = flashim.load("xoftr", preset, device="cuda")
    with torch.inference_mode():
        k0, k1, c = model.match_prepared(gray0, gray1)
    assert len(c) > 0.8 * len(rc)
    da = torch.cdist(rk0.float(), k0.float(), compute_mode="donot_use_mm_for_euclid_dist")
    db = torch.cdist(rk1.float(), k1.float(), compute_mode="donot_use_mm_for_euclid_dist")
    agree = float(((da < 2.0) & (db < 2.0)).any(dim=1).float().mean())
    assert agree >= min_agree, f"{preset} agree={agree:.4f} < {min_agree}"

