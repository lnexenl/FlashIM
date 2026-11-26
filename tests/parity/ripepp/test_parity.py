"""FlashIM's RIPE++ against fraunhoferhhi/RIPEpp (loaded from an installed package, ``$FLASHIM_RIPEPP_ROOT``, or
fetched on demand via ``_reference.py`` — never vendored because Fraunhofer's academic license does not allow
redistribution), plus RIPE++ LightGlue parity against vendored ``tests/oracles/lightglue``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import flashim
from flashim.models.dedode.model import MEAN, STD
from flashim.models.ripepp import weights
from flashim.models.weights_util import find_weights

from ._reference import import_upstream_classes

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "assets"
PAIR = (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg")
LIGHTGLUE_ORACLE = REPO / "tests" / "oracles" / "lightglue"


def _build_upstream(variant: str):
    VGG, KeypointSampler, UpstreamRIPEPP = import_upstream_classes()
    try:
        state_dict = weights.load_state_dict(variant=variant)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    backbone = VGG(nchannels=3, pretrained=False, use_instance_norm=False, mode="dect")
    sampler = KeypointSampler(
        window_size=8, threshold=0.5, top_k=2048, subpixel_sampling=True, sub_pixel_temp=0.5, nms_size=3
    )
    model = UpstreamRIPEPP(
        net=backbone, keypoint_sampler=sampler, window_size=8, non_linearity_dect=torch.nn.Identity(), descriptor_dim=256
    )
    model.load_state_dict(state_dict, strict=True)
    return model.cuda().eval()


def _imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [("ripepp", "default"), ("ripepp-aachen", "aachen"), ("ripepp-scared", "scared")],
)
def test_fp32_matches_oracle(model_name, variant, monkeypatch):
    from flashim.models.romav2.model import load_image

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    oracle = _build_upstream(variant)
    model = flashim.load(model_name, "fp32")

    for path in PAIR:
        image = load_image(str(path), torch.device("cuda"))
        image = torch.nn.functional.interpolate(image, size=(512, 664), mode="bilinear", align_corners=False, antialias=True)
        with torch.inference_mode():
            ref_k, ref_d, ref_s = oracle.detectAndCompute(
                _imagenet_normalize(image), threshold=0.5, top_k=1500, nms_size=3, subpixel_sampling=True
            )
            kpts, scores, desc = model.detect(image, 1500)

        # Upstream returns [0.5, W - 0.5]; FlashIM SparseExtractor uses integer = pixel center (upstream - 0.5).
        ref_k = ref_k[0] - 0.5
        ref_d = ref_d[0]
        ref_s = ref_s[0]

        # Compare keypoints as sets via nearest-neighbour distance (a few borderline peaks may tie/flip).
        dists = torch.cdist(kpts.float(), ref_k.float(), compute_mode="donot_use_mm_for_euclid_dist")
        min_dist, nn_idx = dists.min(dim=1)
        matched = min_dist < 1e-3
        assert matched.sum().item() >= 0.98 * len(ref_k)
        score_diff = (scores[matched] - ref_s[nn_idx[matched]]).abs().max().item()
        assert score_diff < 2e-3
        cos = (desc[matched] * ref_d[nn_idx[matched]]).sum(dim=-1)
        assert cos.min().item() > 0.998


def test_ripepp_lightglue_fp32_matches_oracle(monkeypatch):
    if not (LIGHTGLUE_ORACLE / "lightglue").is_dir():
        pytest.skip("needs vendored lightglue oracle")
    try:
        lg_ckpt = find_weights(weights.LIGHTGLUE_CHECKPOINT, env="FLASHIM_RIPEPP_LIGHTGLUE_CHECKPOINT")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    sys.path.insert(0, str(LIGHTGLUE_ORACLE))
    try:
        from lightglue.lightglue import LightGlue as UpstreamLightGlue
    finally:
        sys.path.remove(str(LIGHTGLUE_ORACLE))

    pipeline = flashim.load("ripepp-lightglue", "fp32")
    with torch.inference_mode():
        fa = pipeline.extract(str(PAIR[0]), resize=1024, top_k=2048)
        fb = pipeline.extract(str(PAIR[1]), resize=1024, top_k=2048)
        result = pipeline.match_features(fa, fb)

    oracle = UpstreamLightGlue(features=None, input_dim=256, depth_confidence=-1, width_confidence=-1, flash=False)
    raw = torch.load(lg_ckpt, map_location="cpu", weights_only=True)
    state = {k[len("matcher.") :]: v for k, v in raw["model"].items() if k.startswith("matcher.")}
    missing, unexpected = oracle.load_state_dict(state, strict=False)
    assert not [k for k in missing if not k.startswith("confidence_thresholds")] and not unexpected
    oracle = oracle.to(pipeline.device).eval()

    with torch.inference_mode():
        size = lambda f: torch.tensor([[f.image_size[1], f.image_size[0]]], device=pipeline.device)  # (w, h)
        # Upstream ripe++lightglue.tar expects [0.5, W - 0.5] keypoints (+0.5 over FlashIM pixel-center coords).
        pred = oracle(
            {
                "image0": {"keypoints": fa.keypoints + 0.5, "descriptors": fa.descriptors, "image_size": size(fa)},
                "image1": {"keypoints": fb.keypoints + 0.5, "descriptors": fb.descriptors, "image_size": size(fb)},
            }
        )

    expected = set(map(tuple, pred["matches"][0].tolist()))
    got = set(zip(result.indices_A.tolist(), result.indices_B.tolist()))
    assert len(expected) > 100 and got == expected
