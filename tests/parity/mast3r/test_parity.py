"""FlashIM's MASt3R against naver/mast3r + naver/dust3r + naver/croco (vendored in ``tests/oracles/mast3r``)."""

from __future__ import annotations

import sys
import types

import pytest
import torch

import flashim
from flashim.models.mast3r import weights
from flashim.models.weights_util import find_weights
from flashim.paths import ASSETS, ORACLES

pytestmark = [pytest.mark.oracle, pytest.mark.weights, pytest.mark.cuda, pytest.mark.slow]

ORACLE = ORACLES / "mast3r"
PAIRS = [
    pytest.param((ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg"), id="sacre_coeur"),
    pytest.param((ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg"), id="toronto"),
]


@pytest.fixture(scope="module")
def upstream():
    if not (ORACLE / "mast3r").exists():
        pytest.skip("needs vendored mast3r oracle")
    # fast_nn.py imports scipy.spatial.KDTree only for CPU fallback; GPU uses cdistMatcher.
    added_scipy = False
    if "scipy" not in sys.modules:
        try:
            import scipy.spatial  # noqa: F401
        except ImportError:
            scipy_mod = types.ModuleType("scipy")
            spatial_mod = types.ModuleType("scipy.spatial")
            spatial_mod.KDTree = None
            scipy_mod.spatial = spatial_mod
            sys.modules["scipy"] = scipy_mod
            sys.modules["scipy.spatial"] = spatial_mod
            added_scipy = True

    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    sys.path.insert(0, str(ORACLE))
    try:
        import importlib

        from mast3r.fast_nn import fast_reciprocal_NNs
        from mast3r.model import AsymmetricMASt3R

        # Must follow mast3r.model so mast3r.utils.path_to_dust3r adds dust3r/ to sys.path.
        load_images = importlib.import_module("dust3r.utils.image").load_images
    finally:
        sys.path.remove(str(ORACLE))
        # models/croco.py sets allow_tf32 = True on import; restore immediately.
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        if added_scipy:
            sys.modules.pop("scipy.spatial", None)
            sys.modules.pop("scipy", None)

    ckpt_path = find_weights(weights.CHECKPOINT, env="FLASHIM_MAST3R_CHECKPOINT")
    oracle_model = AsymmetricMASt3R.from_pretrained(str(ckpt_path)).cuda().eval()
    return oracle_model, load_images, fast_reciprocal_NNs


@pytest.mark.parametrize("pair", PAIRS)
def test_fp32_matches_oracle(upstream, pair, monkeypatch):
    """FP32 FlashIM MASt3R reproduces upstream pts3d, conf, desc, desc_conf and fast_reciprocal_NNs matches."""
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    oracle_model, load_images, fast_reciprocal_NNs = upstream
    views = load_images([str(p) for p in pair], size=512, verbose=False)
    v1 = {
        "img": views[0]["img"].cuda(),
        "true_shape": torch.from_numpy(views[0]["true_shape"]),
        "idx": [views[0]["idx"]],
        "instance": [views[0]["instance"]],
    }
    v2 = {
        "img": views[1]["img"].cuda(),
        "true_shape": torch.from_numpy(views[1]["true_shape"]),
        "idx": [views[1]["idx"]],
        "instance": [views[1]["instance"]],
    }

    model = flashim.load("mast3r", "fp32", device="cuda")
    with torch.inference_mode():
        ref1, ref2 = oracle_model(v1, v2)
        out1, out2 = model(v1["img"], v2["img"])

        ref_xy1, ref_xy2 = fast_reciprocal_NNs(
            ref1["desc"].squeeze(0), ref2["desc"].squeeze(0),
            subsample_or_initxy1=8, device="cuda", dist="dot", block_size=2**13,
        )
        h1, w1 = int(v1["true_shape"][0, 0]), int(v1["true_shape"][0, 1])
        h2, w2 = int(v2["true_shape"][0, 0]), int(v2["true_shape"][0, 1])
        valid = (
            (ref_xy1[:, 0] >= 3) & (ref_xy1[:, 0] < w1 - 3) & (ref_xy1[:, 1] >= 3) & (ref_xy1[:, 1] < h1 - 3)
            & (ref_xy2[:, 0] >= 3) & (ref_xy2[:, 0] < w2 - 3) & (ref_xy2[:, 1] >= 3) & (ref_xy2[:, 1] < h2 - 3)
        )
        ref_xy1, ref_xy2 = ref_xy1[valid], ref_xy2[valid]

        our_xy1, our_xy2, our_conf = model.match_prepared(v1["img"], v2["img"])

    # Dense tensor parity: pts3d, conf, desc, desc_conf
    torch.testing.assert_close(out1["pts3d"], ref1["pts3d"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out2["pts3d_in_other_view"], ref2["pts3d_in_other_view"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out1["conf"], ref1["conf"], atol=5e-4, rtol=1e-4)
    torch.testing.assert_close(out2["conf"], ref2["conf"], atol=5e-4, rtol=1e-4)
    torch.testing.assert_close(out1["desc"], ref1["desc"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out2["desc"], ref2["desc"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out1["desc_conf"], ref1["desc_conf"], atol=5e-4, rtol=1e-4)
    torch.testing.assert_close(out2["desc_conf"], ref2["desc_conf"], atol=5e-4, rtol=1e-4)

    # Reciprocal NN match agreement (exact pixel coordinates on prepared inputs)
    assert len(our_conf) == len(our_xy1) > 100
    ref_set = {(int(a[0]), int(a[1]), int(b[0]), int(b[1])) for a, b in zip(ref_xy1, ref_xy2)}
    our_set = {
        (int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        for a, b in zip(our_xy1.cpu().tolist(), our_xy2.cpu().tolist())
    }
    overlap = len(ref_set & our_set) / max(len(ref_set), 1)
    assert overlap >= 0.99, f"exact match overlap {overlap:.4f} ({len(ref_set & our_set)}/{len(ref_set)})"


def test_match_api_original_coords(upstream):
    """End-to-end ``model.match(path_a, path_b)`` returns valid Matches in original image coordinates."""
    from PIL import Image

    pair = (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg")
    w1, h1 = Image.open(pair[0]).size
    w2, h2 = Image.open(pair[1]).size
    model = flashim.load("mast3r", "fp32", device="cuda")
    m = model.match(pair[0], pair[1])
    assert len(m) > 100
    assert m.keypoints_A.shape == (len(m), 2)
    assert m.keypoints_B.shape == (len(m), 2)
    assert m.confidence.shape == (len(m),)
    assert (m.keypoints_A[:, 0] >= -0.5).all() and (m.keypoints_A[:, 0] < w1).all()
    assert (m.keypoints_A[:, 1] >= -0.5).all() and (m.keypoints_A[:, 1] < h1).all()
    assert (m.keypoints_B[:, 0] >= -0.5).all() and (m.keypoints_B[:, 0] < w2).all()
    assert (m.keypoints_B[:, 1] >= -0.5).all() and (m.keypoints_B[:, 1] < h2).all()
    assert torch.isfinite(m.confidence).all() and (m.confidence > 0).all()

    m_capped = model.match(pair[0], pair[1], max_matches=50)
    assert len(m_capped) == 50
