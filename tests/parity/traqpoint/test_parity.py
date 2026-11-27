"""FlashIM's native TraqPoint against the reference implementation (xiaomi-research/traqpoint in tests/oracles/traqpoint)."""

import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ...conftest import MEGADEPTH_IMAGES, ORACLES

pytestmark = [pytest.mark.cuda, pytest.mark.slow]

TRAQPOINT_ROOT = ORACLES / "traqpoint"
IMAGES = [MEGADEPTH_IMAGES / "505127_40b48ea397_o.jpg", MEGADEPTH_IMAGES / "505128_561e05e146_o.jpg"]

DEFAULT_CONFIG = {
    "activation": "relu",
    "block_dims": [8, 16, 32, 64],
    "d_model": 256,
    "detection_threshold": 0.0,
    "device": "cuda",
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "enc_n_points": 8,
    "hidden_dim": 256,
    "lr_backbone": 0,
    "nhead": 8,
    "num_encoder_layers": 4,
    "num_feature_levels": 4,
    "top_k": 4096,
    "train_detector": False,
}


@pytest.fixture(scope="module")
def checkpoint():
    from flashim.models.traqpoint.weights import find_checkpoint

    try:
        return find_checkpoint()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def oracle(checkpoint):
    if not (TRAQPOINT_ROOT / "traqpoint" / "traqpoint.py").exists():
        pytest.skip(f"needs the TraqPoint reference repository at {TRAQPOINT_ROOT}")
    sys.path.insert(0, str(TRAQPOINT_ROOT))
    try:
        import dinov3_convnext
        from traqpoint.models import backbone_dinov3_conv
        from traqpoint.traqpoint import build
    finally:
        sys.path.remove(str(TRAQPOINT_ROOT))

    # The pretrained traqpoint_best.pth already stores the fine-tuned DINOv3 ConvNeXt-B weights, so bypass the
    # upstream backbone's local third_party/ check and initial DINOv3 checkpoint load during build().
    full_sd = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    dino_prefix = "descriptor.backbone.0.dino."
    dino_sd = {k[len(dino_prefix):]: v for k, v in full_sd.items() if k.startswith(dino_prefix)}

    real_torch_load = torch.load

    def _fake_load(path, *args, **kwargs):
        if isinstance(path, str) and "dinov3_convnext_base" in path:
            return dino_sd
        return real_torch_load(path, *args, **kwargs)

    with (
        patch.object(backbone_dinov3_conv, "_require_path", lambda *a, **kw: None),
        patch.object(torch.hub, "load", lambda *a, **kw: dinov3_convnext.get_convnext_arch("convnext_base")()),
        patch.object(torch, "load", _fake_load),
    ):
        model = build(config=dict(DEFAULT_CONFIG), weights=str(checkpoint)).cuda()
    model.eval()
    return model


def _images():
    if not IMAGES:
        pytest.skip("needs MegaDepth images under assets/megadepth_0015")
    from PIL import Image

    out = []
    for path in IMAGES:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        image = image.crop((0, 0, w // 32 * 32, h // 32 * 32)).resize((640, 480))
        out.append(torch.from_numpy(np.asarray(image)).permute(2, 0, 1)[None].float().cuda() / 255)
    return out


@torch.inference_mode()
def test_fp32_matches_reference(oracle):
    from flashim.models.traqpoint import TraqPoint

    prev_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        ours = TraqPoint.from_pretrained(device="cuda")
        for image in _images():
            ref_desc, ref_mat, _, _, ref_scores = oracle(image)
            desc, scores = ours(image)
            torch.testing.assert_close(scores, ref_scores, atol=1e-6, rtol=1e-6)
            assert ((desc - ref_desc).norm() / ref_desc.norm()) < 1e-4
            torch.testing.assert_close(ours.matchability(desc), ref_mat, atol=1e-4, rtol=1e-4)
            ref = oracle.extract(image)[0]
            out = ours.extract(image)
            assert (out.keypoints[0] - ref["keypoints"]).abs().max() < 1e-3
            assert (out.scores[0] - ref["scores"]).abs().max() < 1e-5
            assert (out.descriptors[0] * ref["descriptors"]).sum(-1).min() > 0.9999
    finally:
        torch.backends.cudnn.allow_tf32 = prev_tf32


@torch.inference_mode()
@pytest.mark.parametrize("preset", ["bf16", "int8", "fp8"])
def test_reduced_precision_tracks_fp32(checkpoint, preset):
    import flashim

    reference = flashim.load("traqpoint", preset="fp32")
    model = flashim.build_config(flashim.resolve("traqpoint", preset))
    if preset == "fp8":
        assert model.quant_report.fused_blocks == ["transformer.encoder"]
    for image in _images():
        a, b = reference.describe(image), model.describe(image).float()
        cosine = F.cosine_similarity(a, b, dim=1)
        assert cosine.mean() > 0.98, f"{preset}: mean descriptor cosine {cosine.mean():.4f}"
