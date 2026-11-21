"""FlashIM's native RoMa (v1) against the vendored Parskatt/RoMa reference implementation (same process, same inputs).

FP32 with ``local_corr:reference`` must be bit-identical to upstream ``roma_outdoor`` / ``roma_indoor`` on dense warp,
certainty, and fixed-seed sampling (with ``kde_density:reference``). Default Triton ``local_corr`` is checked against
the reference within tight tolerances; ``bf16`` is checked against ``fp32``.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest
import torch
from PIL import Image

import flashim
from flashim import ops
from flashim.models.roma import weights
from flashim.models.roma.model import prepare_pair
from flashim.models.romav2.geometry import imagenet_normalize
from flashim.paths import ORACLES

from ...conftest import ASSETS

pytestmark = [pytest.mark.cuda, pytest.mark.weights, pytest.mark.oracle, pytest.mark.slow]

ORACLE = ORACLES / "roma"
PAIR = (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")


@pytest.fixture(scope="module")
def romatch_pkg():
    if not (ORACLE / "romatch").exists():
        pytest.skip("vendored RoMa oracle not present")
    if "loguru" not in sys.modules:
        stub = types.ModuleType("loguru")
        stub.logger = logging.getLogger("romatch")
        sys.modules["loguru"] = stub
    sys.path.insert(0, str(ORACLE))
    try:
        import romatch
        import romatch.models.encoders as r_enc
        import romatch.models.matcher as r_mat
        import romatch.models.transformer as r_tr
        import romatch.utils.utils as r_ut
    finally:
        sys.path.remove(str(ORACLE))

    # Upstream's get_autocast_params unconditionally enables autocast on CUDA; disable it for true FP32 parity.
    def _no_autocast(device=None, enabled=False, dtype=None):
        return "cuda", False, torch.float32

    for mod in (r_enc, r_mat, r_tr, r_ut):
        mod.get_autocast_params = _no_autocast
    return romatch


def _build_oracle(romatch_pkg, variant: str):
    factory = romatch_pkg.roma_outdoor if variant == "outdoor" else romatch_pkg.roma_indoor
    oracle = factory(
        device="cuda",
        weights=weights.load_state_dict(variant),
        dinov2_weights=weights.load_dinov2(),
        amp_dtype=torch.float32,
        use_custom_corr=False,
    ).eval()
    oracle.encoder.dinov2_vitl14[0].float().cuda()
    oracle.encoder.amp = False
    oracle.encoder.amp_dtype = torch.float32
    return oracle


@pytest.mark.parametrize(("name", "variant"), [("roma", "outdoor"), ("roma-indoor", "indoor")])
def test_fp32_bit_exact_against_oracle(romatch_pkg, name, variant, fp32_highest, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    oracle = _build_oracle(romatch_pkg, variant)
    model = flashim.load(name, "fp32", device="cuda")
    a_lr, b_lr, a_hr, b_hr, _, _ = prepare_pair(*PAIR, model.config, torch.device("cuda"))
    an_lr, bn_lr = imagenet_normalize(a_lr), imagenet_normalize(b_lr)
    an_hr, bn_hr = imagenet_normalize(a_hr), imagenet_normalize(b_hr)

    with torch.inference_mode():
        ref_warp, ref_cert = oracle.match(an_lr, bn_lr, im_A_high_res=an_hr, im_B_high_res=bn_hr, device="cuda")
        w = ref_warp.shape[2] // 2
        ref_ab, ref_ba = ref_warp[:, :, :w, 2:], ref_warp[:, :, w:, :2]
        ref_cab, ref_cba = ref_cert[:, :, :w], ref_cert[:, :, w:]

        # 1) Exact match with reference local_corr
        ops.configure("local_corr:reference")
        out_ref = model.forward_normalized(an_lr, bn_lr, an_hr, bn_hr)
        assert torch.equal(out_ref.warp_AB, ref_ab)
        assert torch.equal(out_ref.warp_BA, ref_ba)
        assert torch.equal(out_ref.overlap_AB[..., 0], ref_cab)
        assert torch.equal(out_ref.overlap_BA[..., 0], ref_cba)

        # 2) Default Triton local_corr stays very close
        ops.configure(None)
        out_def = model.forward_normalized(an_lr, bn_lr, an_hr, bn_hr)
        assert (out_def.warp_AB - ref_ab).abs().max() < 2e-3
        assert (out_def.warp_BA - ref_ba).abs().max() < 2e-3
        assert (out_def.overlap_AB[..., 0] - ref_cab).abs().max() < 2e-3
        assert (out_def.overlap_BA[..., 0] - ref_cba).abs().max() < 2e-3

        # 3) Sampling parity: bit-exact with reference KDE and the same RNG seed
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        ref_matches, ref_conf = oracle.sample(ref_warp[0], ref_cert[0], num=5000)

        model.fast_sampling = False
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        our_matches, our_conf = model.sample(out_ref, num=5000)
        assert torch.equal(our_matches, ref_matches)
        assert torch.equal(our_conf, ref_conf)


@pytest.mark.parametrize("preset", ["bf16", "int8", "fp8"])
def test_match_api_and_reduced_precision_tracks_fp32(preset, fp32_highest):
    fp32 = flashim.load("roma", "fp32", device="cuda")
    model = flashim.load("roma", preset, device="cuda")

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
