"""FlashIM's MINIMA variants (``minima-loftr``, ``minima-roma``) against upstream MINIMA code (FP32).

- ``minima-loftr`` is checked against the vendored ``LSXI7/LoFTR_minima`` inference code (``temp_bug_fix=True``,
  matching ``MINIMA/load_model.py``).
- ``minima-roma`` is checked against the vendored ``romatch`` reference in ``tests/oracles/roma/`` (bit-identical to
  ``LSXI7/RoMa_minima`` with ``local_corr:reference`` and ``kde_density:reference``).
"""

from __future__ import annotations

import copy
import logging
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import flashim
from flashim import ops
from flashim.models import sparse
from flashim.models.loftr import weights as loftr_weights
from flashim.models.roma import weights as roma_weights
from flashim.models.roma.model import prepare_pair
from flashim.models.romav2.geometry import imagenet_normalize
from flashim.models.romav2.model import load_image
from flashim.paths import ORACLES

from ...conftest import ASSETS

pytestmark = [pytest.mark.cuda, pytest.mark.weights, pytest.mark.oracle, pytest.mark.slow]

PAIR = (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")
LOFTR_PAIR = (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg")
MINIMA_ORACLE = ORACLES / "minima"
ROMA_ORACLE = ORACLES / "roma"


@pytest.fixture(scope="module")
def loftr_minima_oracle():
    if not (MINIMA_ORACLE / "loftr").exists():
        pytest.skip("vendored MINIMA LoFTR oracle not present")
    pytest.importorskip("kornia")
    import kornia.geometry

    # LoFTR_minima imports ``kornia.utils.grid.create_meshgrid`` (kornia <=0.6); shim for newer kornia.
    if "kornia.utils.grid" not in sys.modules:
        grid_mod = types.ModuleType("kornia.utils.grid")
        grid_mod.create_meshgrid = kornia.geometry.create_meshgrid
        sys.modules["kornia.utils.grid"] = grid_mod
    sys.path.insert(0, str(MINIMA_ORACLE))
    try:
        from loftr import LoFTR as OracleLoFTR
        from loftr import default_cfg
    finally:
        sys.path.remove(str(MINIMA_ORACLE))

    cfg = copy.deepcopy(default_cfg)
    cfg["coarse"]["temp_bug_fix"] = True  # MINIMA/load_model.py
    cfg["match_coarse"]["thr"] = 0.2
    model = OracleLoFTR(config=cfg)
    state = loftr_weights.load_state_dict("minima")
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


@pytest.fixture(scope="module")
def roma_minima_oracle():
    if not (ROMA_ORACLE / "romatch").exists():
        pytest.skip("vendored RoMa oracle not present")
    stubbed_loguru = "loguru" not in sys.modules
    if stubbed_loguru:
        stub = types.ModuleType("loguru")
        stub.logger = logging.getLogger("romatch")
        stub._Logger = object
        sys.modules["loguru"] = stub
    sys.path.insert(0, str(ROMA_ORACLE))
    try:
        import romatch
        import romatch.models.encoders as r_enc
        import romatch.models.matcher as r_mat
        import romatch.models.transformer as r_tr
        import romatch.utils.utils as r_ut
    finally:
        sys.path.remove(str(ROMA_ORACLE))
        if stubbed_loguru:
            sys.modules.pop("loguru", None)

    def _no_autocast(device=None, enabled=False, dtype=None):
        return "cuda", False, torch.float32

    for mod in (r_enc, r_mat, r_tr, r_ut):
        mod.get_autocast_params = _no_autocast

    oracle = romatch.roma_outdoor(
        device="cuda",
        weights=roma_weights.load_state_dict("minima"),
        dinov2_weights=roma_weights.load_dinov2(),
        amp_dtype=torch.float32,
        use_custom_corr=False,
    ).eval()
    oracle.encoder.dinov2_vitl14[0].float().cuda()
    oracle.encoder.amp = False
    oracle.encoder.amp_dtype = torch.float32
    return oracle


def test_minima_loftr_fp32_matches_oracle(loftr_minima_oracle, fp32_highest, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    grays = []
    for path in LOFTR_PAIR:
        img = load_image(path, torch.device("cuda"))
        img = F.interpolate(img, size=(480, 640), mode="bilinear", align_corners=False, antialias=True)
        grays.append(sparse.rgb_to_gray(img))

    # 1) Unfolded FP32 with reference dual_softmax_mutual: exact architecture/math parity.
    raw = flashim.load("minima-loftr", "fp32", device="cuda", overrides={"optimize.passes": []})
    with torch.inference_mode():
        batch = {"image0": grays[0], "image1": grays[1]}
        loftr_minima_oracle(batch)
        ref_k0, ref_k1, ref_c = batch["mkpts0_f"], batch["mkpts1_f"], batch["mconf"]
        assert len(ref_c) > 50

        ops.configure("dual_softmax_mutual:reference")
        try:
            k0_raw, k1_raw, c_raw = raw.match_prepared(*grays)
        finally:
            ops.configure(None)

    assert len(c_raw) == len(ref_c)
    assert (k0_raw - ref_k0).abs().max().item() < 1e-4
    assert (k1_raw - ref_k1).abs().max().item() < 1e-4
    assert (c_raw - ref_c).abs().max().item() < 1e-5

    # 2) Default fp32 preset (BN fold + channels_last + Triton dual_softmax_mutual).
    model = flashim.load("minima-loftr", "fp32", device="cuda")
    with torch.inference_mode():
        k0, k1, conf = model.match_prepared(*grays)

    assert abs(len(conf) - len(ref_c)) <= 0.02 * len(ref_c) + 2
    ours = {tuple(a): (tuple(b), c) for a, b, c in zip(k0.tolist(), k1.tolist(), conf.tolist())}
    common = 0
    for a, b, c in zip(ref_k0.tolist(), ref_k1.tolist(), ref_c.tolist()):
        if tuple(a) in ours:
            common += 1
            assert max(abs(ours[tuple(a)][0][i] - b[i]) for i in (0, 1)) < 0.05
            assert abs(ours[tuple(a)][1] - c) < 5e-3
    assert common >= 0.97 * len(ref_c)


def test_minima_roma_fp32_bit_exact_against_oracle(roma_minima_oracle, fp32_highest, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    model = flashim.load("minima-roma", "fp32", device="cuda")
    a_lr, b_lr, a_hr, b_hr, _, _ = prepare_pair(*PAIR, model.config, torch.device("cuda"))
    an_lr, bn_lr = imagenet_normalize(a_lr), imagenet_normalize(b_lr)
    an_hr, bn_hr = imagenet_normalize(a_hr), imagenet_normalize(b_hr)

    with torch.inference_mode():
        ref_warp, ref_cert = roma_minima_oracle.match(
            an_lr, bn_lr, im_A_high_res=an_hr, im_B_high_res=bn_hr, device="cuda"
        )
        w = ref_warp.shape[2] // 2
        ref_ab, ref_ba = ref_warp[:, :, :w, 2:], ref_warp[:, :, w:, :2]
        ref_cab, ref_cba = ref_cert[:, :, :w], ref_cert[:, :, w:]

        # 1) Exact match with reference local_corr
        ops.configure("local_corr:reference")
        try:
            out_ref = model.forward_normalized(an_lr, bn_lr, an_hr, bn_hr)
        finally:
            ops.configure(None)
        assert torch.equal(out_ref.warp_AB, ref_ab)
        assert torch.equal(out_ref.warp_BA, ref_ba)
        assert torch.equal(out_ref.overlap_AB[..., 0], ref_cab)
        assert torch.equal(out_ref.overlap_BA[..., 0], ref_cba)

        # 2) Default Triton local_corr stays very close
        out_def = model.forward_normalized(an_lr, bn_lr, an_hr, bn_hr)
        assert (out_def.warp_AB - ref_ab).abs().max() < 5e-3 and (out_def.warp_AB - ref_ab).abs().mean() < 1e-4
        assert (out_def.warp_BA - ref_ba).abs().max() < 5e-3 and (out_def.warp_BA - ref_ba).abs().mean() < 1e-4
        assert (out_def.overlap_AB[..., 0] - ref_cab).abs().max() < 5e-3
        assert (out_def.overlap_BA[..., 0] - ref_cba).abs().max() < 5e-3

        # 3) Sampling parity: bit-exact with reference KDE and the same RNG seed
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        ref_matches, ref_conf = roma_minima_oracle.sample(ref_warp[0], ref_cert[0], num=5000)

        model.fast_sampling = False
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        our_matches, our_conf = model.sample(out_ref, num=5000)
        assert torch.equal(our_matches, ref_matches)
        assert torch.equal(our_conf, ref_conf)


@pytest.mark.parametrize("preset", ["bf16", "int8", "fp8"])
def test_minima_roma_reduced_precision_tracks_fp32(preset, fp32_highest):
    fp32 = flashim.load("minima-roma", "fp32", device="cuda")
    model = flashim.load("minima-roma", preset, device="cuda")

    ref = fp32.match_dense(*PAIR)
    out = model.match(*PAIR, max_matches=2000)
    assert out.dense is not None
    assert out.keypoints_A.shape == out.keypoints_B.shape == (2000, 2)
    assert out.confidence.shape == (2000,)

    (w_a, h_a), (w_b, h_b) = Image.open(PAIR[0]).size, Image.open(PAIR[1]).size
    for kp, (w, h) in ((out.keypoints_A, (w_a, h_a)), (out.keypoints_B, (w_b, h_b))):
        assert (kp[:, 0] >= 0).all() and (kp[:, 0] <= w).all()
        assert (kp[:, 1] >= 0).all() and (kp[:, 1] <= h).all()

    h, w = ref.warp_AB.shape[1:3]
    confident = ref.overlap_AB[0, ..., 0] > 0.5
    err_px = (
        (out.dense.warp_AB[0][confident] - ref.warp_AB[0][confident]) * torch.tensor([w / 2, h / 2], device="cuda")
    ).norm(dim=-1)
    assert err_px.median().item() < 0.5, f"median {preset} vs FP32 warp error {err_px.median().item():.3f}px"


@pytest.mark.parametrize("preset", ["bf16", "int8", "fp8"])
def test_minima_loftr_reduced_precision_tracks_fp32(preset, fp32_highest):
    fp32 = flashim.load("minima-loftr", "fp32", device="cuda")
    model = flashim.load("minima-loftr", preset, device="cuda")

    ref = fp32.match(*LOFTR_PAIR, resize=840)
    out = model.match(*LOFTR_PAIR, resize=840)
    assert len(ref.confidence) > 100 and len(out.confidence) > 100

    da = torch.cdist(ref.keypoints_A.float(), out.keypoints_A.float(), compute_mode="donot_use_mm_for_euclid_dist")
    db = torch.cdist(ref.keypoints_B.float(), out.keypoints_B.float(), compute_mode="donot_use_mm_for_euclid_dist")
    agree = ((da < 2.0) & (db < 2.0)).any(1).float().mean().item()
    assert agree > 0.75, f"minima-loftr {preset} match agreement vs fp32 too low: {agree:.4f}"
