"""FlashIM's native RaCo + ALIKED against fabio-sim/LightGlue-ONNX's PyTorch modules (vendored in
``tests/oracles/raco``, itself derived from cvg/RaCo). Needs CUDA and the RaCo/ALIKED checkpoints."""

import sys

import pytest
import torch
import torch.nn.functional as F

from flashim.paths import ASSETS, ORACLES

pytestmark = [pytest.mark.cuda, pytest.mark.slow]
ROOT = ORACLES / "raco"
IMAGES = [ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg"]


@pytest.fixture(scope="module")
def oracle():
    if not (ROOT / "lightglue_dynamo").exists() or not all(p.exists() for p in IMAGES):
        pytest.skip("needs the sacre_coeur images")
    from flashim.models.raco.weights import find_checkpoint

    try:
        checkpoint = find_checkpoint()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    sys.path.insert(0, str(ROOT))
    try:
        from lightglue_dynamo.models.raco import RaCo as Oracle
    finally:
        sys.path.remove(str(ROOT))
    model = Oracle(num_keypoints=1024, weights=None).cuda().eval()
    model._load_weights(str(checkpoint))
    return model


def _images():
    from flashim.models.romav2.model import load_image

    if not all(p.exists() for p in IMAGES):
        pytest.skip(f"needs the LightGlue-ONNX assets under {ROOT}")
    return [F.interpolate(load_image(p, "cuda"), size=(1024, 1024), mode="bilinear", align_corners=False) for p in IMAGES]


@torch.inference_mode()
def test_fp32_keypoints_and_ranker_scores_match_reference(oracle):
    from flashim.models.raco import RaCo, RaCoConfig

    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ours = RaCo.from_pretrained(RaCoConfig(num_keypoints=1024), device="cuda")
        for image in _images():
            ref_kp, _, ref_scores = oracle(image)
            keypoints, scores = ours.extract_keypoints(image)
            a, b = (set(map(tuple, k[0].round().int().tolist())) for k in (ref_kp, keypoints))
            assert len(a & b) / len(a) > 0.999
            assert (keypoints - ref_kp).abs().max() < 1e-3
            assert (scores - ref_scores).abs().max() < 0.05
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


@torch.inference_mode()
def test_bf16_ranker_keeps_the_selected_keypoints():
    from flashim.models.raco import RaCo
    from flashim.models.raco.fuse import fold_batchnorm

    reference, reduced = (RaCo.from_pretrained(device="cuda") for _ in range(2))
    fold_batchnorm(reference), fold_batchnorm(reduced)
    reduced.set_compute_dtype(torch.bfloat16)
    for image in _images():
        (a, _), (b, _) = reference.extract_keypoints(image), reduced.extract_keypoints(image)
        sa, sb = (set(map(tuple, k[0].round().int().tolist())) for k in (a, b))
        assert len(sa & sb) / len(sa) > 0.97
