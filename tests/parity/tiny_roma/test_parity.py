"""FlashIM's Tiny-RoMa against Parskatt/RoMa (vendored in ``tests/oracles/tiny_roma`` + ``tests/oracles/xfeat``)."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn.functional as F

import flashim
from flashim.models.romav2.model import load_image
from flashim.models.tiny_roma import TinyRoMa, weights
from flashim.paths import ASSETS, ORACLES
from flashim.types import DenseMatchResult

pytestmark = [pytest.mark.oracle, pytest.mark.weights, pytest.mark.cuda]

ORACLE_TINY = ORACLES / "tiny_roma"
ORACLE_XFEAT = ORACLES / "xfeat"
SACRE_PAIR = (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg")
TORONTO_PAIR = (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")


@pytest.fixture(scope="module")
def oracle():
    if not (ORACLE_TINY / "romatch" / "models" / "tiny.py").exists():
        pytest.skip("needs vendored tiny_roma oracle")
    sys.path.insert(0, str(ORACLE_TINY))
    sys.path.insert(0, str(ORACLE_XFEAT))
    try:
        from modules.model import XFeatModel
        from romatch.models.tiny import TinyRoMa as OracleTinyRoMa
    finally:
        sys.path.remove(str(ORACLE_XFEAT))
        sys.path.remove(str(ORACLE_TINY))
    state = weights.load_state_dict(map_location="cpu")
    model = OracleTinyRoMa(xfeat=XFeatModel(), freeze_xfeat=False, exact_softmax=False)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def _load_resized(path, size: tuple[int, int] | None = None) -> torch.Tensor:
    im = load_image(str(path), torch.device("cuda"), decoder="pil")
    if size is not None:
        im = F.interpolate(im, size=size, mode="bilinear", align_corners=False)
    return im


def test_fp32_parity_different_sizes(oracle, monkeypatch):
    """Different-aspect-ratio pair (``sacre_coeur1.jpg`` 731x1024 vs ``sacre_coeur2.jpg`` 1024x768): exercises
    the separate-backbone path, ``preprocess_tensor`` flooring to 32, ``corr_volume``, ``pos_embed``, coarse/fine
    refiners, dense upsampling, and KDE-balanced sampling."""
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    im_a = _load_resized(SACRE_PAIR[0])
    im_b = _load_resized(SACRE_PAIR[1])
    unfolded = TinyRoMa.from_pretrained(device="cuda")
    model = flashim.load("tiny-roma", "fp32")
    model.fast_sampling = False

    with torch.inference_mode():
        ref_warp, ref_cert = oracle.match(im_a, im_b, batched=False)
        dense_unfolded = unfolded.match_dense(im_a, im_b)
        dense = model.match_dense(im_a, im_b)

    ref_flow = ref_warp[..., 2:]
    assert dense.warp_AB[0].shape == ref_flow.shape
    assert dense.overlap_AB[0, ..., 0].shape == ref_cert.shape
    assert (dense_unfolded.warp_AB[0] - ref_flow).abs().max().item() < 1e-5
    assert (dense_unfolded.overlap_AB[0, ..., 0] - ref_cert).abs().max().item() < 1e-5

    flow_diff = (dense.warp_AB[0] - ref_flow).abs().max().item()
    cert_diff = (dense.overlap_AB[0, ..., 0] - ref_cert).abs().max().item()
    assert flow_diff < 5e-5, f"max flow abs diff {flow_diff:.3e}"
    assert cert_diff < 5e-5, f"max certainty abs diff {cert_diff:.3e}"

    # Sampling bit-exactness on identical dense inputs (same CUDA RNG state, reference KDE).
    ref_dense = DenseMatchResult(
        warp_AB=ref_flow[None],
        confidence_AB=ref_cert[None, ..., None],
        overlap_AB=ref_cert[None, ..., None],
        precision_AB=None,
    )
    torch.manual_seed(0)
    ref_matches, ref_mcert = oracle.sample(ref_warp, ref_cert, num=2000)
    torch.manual_seed(0)
    our_matches, our_mcert = model.sample(ref_dense, num=2000)
    assert torch.equal(our_matches, ref_matches)
    assert torch.equal(our_mcert, ref_mcert)

    # End-to-end match() sampled pixel coordinates agree with the oracle warp at each sampled pixel.
    model.fast_sampling = True
    m = model.match(im_a, im_b, max_matches=2000, generator=torch.Generator("cuda").manual_seed(0))
    assert len(m) == 2000
    h_a, w_a = im_a.shape[-2:]
    h_b, w_b = im_b.shape[-2:]
    xs = (m.keypoints_A[:, 0] - 0.5).round().long().clamp(0, w_a - 1)
    ys = (m.keypoints_A[:, 1] - 0.5).round().long().clamp(0, h_a - 1)
    ref_kb = oracle._to_pixel_coordinates(ref_flow[ys, xs], h_b, w_b)
    assert (m.keypoints_B - ref_kb).abs().max().item() < 0.05


def test_fp32_parity_same_size(oracle, monkeypatch):
    """Same-size pair (``toronto_A/B`` resized to 480x640): exercises the batched-backbone path."""
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    im_a = _load_resized(TORONTO_PAIR[0], size=(480, 640))
    im_b = _load_resized(TORONTO_PAIR[1], size=(480, 640))
    model = flashim.load("tiny-roma", "fp32")

    with torch.inference_mode():
        ref_warp, ref_cert = oracle.match(im_a, im_b, batched=False)
        dense = model.match_dense(im_a, im_b)

    flow_diff = (dense.warp_AB[0] - ref_warp[..., 2:]).abs().max().item()
    cert_diff = (dense.overlap_AB[0, ..., 0] - ref_cert).abs().max().item()
    assert flow_diff < 1e-4, f"max flow abs diff {flow_diff:.3e}"
    assert cert_diff < 1e-4, f"max certainty abs diff {cert_diff:.3e}"


@pytest.mark.parametrize("preset", ["bf16", "int8", "fp8"])
def test_reduced_precision_tracks_fp32(preset):
    """BF16 / INT8 / FP8 dense warps stay close to FP32 on confident pixels."""
    im_a = _load_resized(SACRE_PAIR[0])
    im_b = _load_resized(SACRE_PAIR[1])
    fp32 = flashim.load("tiny-roma", "fp32")
    reduced = flashim.load("tiny-roma", preset)
    d32 = fp32.match_dense(im_a, im_b)
    dr = reduced.match_dense(im_a, im_b)
    confident = d32.overlap_AB[0, ..., 0] > 0.5
    assert confident.any()
    h, w = im_b.shape[-2:]
    scale = torch.tensor([w / 2, h / 2], device=im_a.device)
    err_px = ((dr.warp_AB[0][confident] - d32.warp_AB[0][confident]) * scale).norm(dim=-1)
    assert err_px.median().item() < 1.0, f"{preset} median warp disagreement {err_px.median().item():.3f}px"
