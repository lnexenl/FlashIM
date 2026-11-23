"""FlashIM's DUSt3R against naver/dust3r + naver/croco (vendored in ``tests/oracles/dust3r``)."""

from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import pytest
import torch

import flashim
from flashim.models.dust3r import weights
from flashim.models.weights_util import find_weights
from flashim.paths import ASSETS, ORACLES

pytestmark = [pytest.mark.oracle, pytest.mark.weights, pytest.mark.cuda, pytest.mark.slow]

ORACLE = ORACLES / "dust3r"
PAIRS = [
    pytest.param((ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg"), id="sacre_coeur"),
    pytest.param((ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg"), id="toronto"),
]


class _ExactL2KDTree:
    """Exact Euclidean NN stand-in for ``scipy.spatial.cKDTree`` when scipy is not installed in the test venv.

    Runs blocked ``torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist')`` in FP32 on CUDA so exact
    difference-squared distances match ``cKDTree`` without needing scipy (and without FP64 slowdown on consumer GPUs).
    """

    def __init__(self, data: np.ndarray) -> None:
        self._db = torch.as_tensor(data, dtype=torch.float32, device="cuda")

    def query(self, pts: np.ndarray, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
        q = torch.as_tensor(pts, dtype=torch.float32, device="cuda")
        nq, nd = q.shape[0], self._db.shape[0]
        block = 4096
        best_d = torch.empty(nq, dtype=torch.float32, device="cuda")
        best_i = torch.empty(nq, dtype=torch.int64, device="cuda")
        for qs in range(0, nq, block):
            qe = min(qs + block, nq)
            qb = q[qs:qe]
            bv = torch.full((qe - qs,), float("inf"), dtype=torch.float32, device="cuda")
            bi = torch.zeros(qe - qs, dtype=torch.int64, device="cuda")
            for ns in range(0, nd, block):
                ne = min(ns + block, nd)
                vals, idx = torch.cdist(qb, self._db[ns:ne], compute_mode="donot_use_mm_for_euclid_dist").min(dim=1)
                better = vals < bv
                bv = torch.where(better, vals, bv)
                bi = torch.where(better, idx + ns, bi)
            best_d[qs:qe] = bv
            best_i[qs:qe] = bi
        return best_d.cpu().numpy(), best_i.cpu().numpy()


@pytest.fixture(scope="module")
def upstream():
    if not (ORACLE / "dust3r").exists():
        pytest.skip("needs vendored dust3r oracle")

    added_scipy = False
    if "scipy" not in sys.modules:
        try:
            import scipy.spatial  # noqa: F401
        except ImportError:
            scipy_mod = types.ModuleType("scipy")
            spatial_mod = types.ModuleType("scipy.spatial")
            spatial_mod.cKDTree = _ExactL2KDTree
            scipy_mod.spatial = spatial_mod
            sys.modules["scipy"] = scipy_mod
            sys.modules["scipy.spatial"] = spatial_mod
            added_scipy = True

    # Evict any dust3r / models modules imported earlier (e.g. by tests/parity/mast3r).
    saved_modules = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k in ("dust3r", "models") or k.startswith(("dust3r.", "models."))
    }
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    sys.path.insert(0, str(ORACLE))
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        from dust3r.utils.image import load_images
    finally:
        sys.path.remove(str(ORACLE))
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        if added_scipy:
            sys.modules.pop("scipy.spatial", None)
            sys.modules.pop("scipy", None)

    ckpt_path = find_weights(weights.CHECKPOINT, env="FLASHIM_DUST3R_CHECKPOINT")
    with torch.serialization.safe_globals([argparse.Namespace]):
        oracle_model = AsymmetricCroCo3DStereo.from_pretrained(str(ckpt_path)).cuda().eval()
    try:
        yield oracle_model, load_images, find_reciprocal_matches, xy_grid
    finally:
        for k in list(sys.modules):
            if k in ("dust3r", "models") or k.startswith(("dust3r.", "models.")):
                sys.modules.pop(k, None)
        sys.modules.update(saved_modules)


@pytest.mark.parametrize("pair", PAIRS)
def test_fp32_matches_oracle(upstream, pair, monkeypatch):
    """FP32 FlashIM DUSt3R reproduces upstream pts3d, conf, and find_reciprocal_matches correspondences."""
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    oracle_model, load_images, find_reciprocal_matches, xy_grid = upstream
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

    model = flashim.load("dust3r", "fp32", device="cuda")
    conf_thr = 3.0
    with torch.inference_mode():
        ref1, ref2 = oracle_model(v1, v2)
        out1, out2 = model(v1["img"], v2["img"])

        # Upstream find_reciprocal_matches (visloc.py convention: conf >= 3.0)
        h1, w1 = int(v1["true_shape"][0, 0]), int(v1["true_shape"][0, 1])
        h2, w2 = int(v2["true_shape"][0, 0]), int(v2["true_shape"][0, 1])
        mask1 = (ref1["conf"][0] >= conf_thr).cpu().numpy()
        mask2 = (ref2["conf"][0] >= conf_thr).cpu().numpy()
        pts2d_1 = xy_grid(w1, h1)[mask1]
        pts2d_2 = xy_grid(w2, h2)[mask2]
        pts3d_1 = ref1["pts3d"][0].cpu().numpy()[mask1]
        pts3d_2 = ref2["pts3d_in_other_view"][0].cpu().numpy()[mask2]
        recip_2, nn2_in_1, _ = find_reciprocal_matches(pts3d_1, pts3d_2)
        ref_xy2 = pts2d_2[recip_2]
        ref_xy1 = pts2d_1[nn2_in_1][recip_2]

        our_xy1, our_xy2, our_conf = model.match_prepared(v1["img"], v2["img"], threshold=conf_thr)

    # Dense tensor parity: pts3d, pts3d_in_other_view, conf
    torch.testing.assert_close(out1["pts3d"], ref1["pts3d"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out2["pts3d_in_other_view"], ref2["pts3d_in_other_view"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out1["conf"], ref1["conf"], atol=5e-4, rtol=1e-4)
    torch.testing.assert_close(out2["conf"], ref2["conf"], atol=5e-4, rtol=1e-4)

    # Reciprocal 3D NN match agreement (exact pixel coordinates on prepared inputs)
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
    model = flashim.load("dust3r", "fp32", device="cuda")
    m = model.match(pair[0], pair[1], threshold=3.0)
    assert len(m) > 100
    assert m.keypoints_A.shape == (len(m), 2)
    assert m.keypoints_B.shape == (len(m), 2)
    assert m.confidence.shape == (len(m),)
    assert (m.keypoints_A[:, 0] >= -0.5).all() and (m.keypoints_A[:, 0] < w1).all()
    assert (m.keypoints_A[:, 1] >= -0.5).all() and (m.keypoints_A[:, 1] < h1).all()
    assert (m.keypoints_B[:, 0] >= -0.5).all() and (m.keypoints_B[:, 0] < w2).all()
    assert (m.keypoints_B[:, 1] >= -0.5).all() and (m.keypoints_B[:, 1] < h2).all()
    assert torch.isfinite(m.confidence).all() and (m.confidence >= 3.0).all()

    m_capped = model.match(pair[0], pair[1], threshold=3.0, max_matches=50)
    assert len(m_capped) == 50
