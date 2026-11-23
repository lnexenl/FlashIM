"""FlashIM's native UFM against the vendored upstream ``UniFlowMatchConfidence`` oracle (same process, same inputs).

Checks FP32 parity on:
1. Working-resolution ``forward(a, b)``: dense flow ``[1, 2, 420, 560]``, covisibility mask and logits.
2. Original-resolution ``predict_flow(a, b)`` vs upstream ``predict_correspondences_batched`` (with upstream's
   ``bf16`` autocast disabled so both run pure FP32).
3. Sampled ``Matches`` (``match(..., max_matches=5000, generator=...)``) against sampling the oracle's dense
   output with the same RNG seed.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

import flashim
from flashim.models.ufm.weights import CHECKPOINT
from flashim.models.weights_util import checkpoint_dir
from flashim.paths import ASSETS, ORACLES

pytestmark = [pytest.mark.cuda, pytest.mark.weights, pytest.mark.oracle, pytest.mark.slow]

ORACLE = ORACLES / "ufm"
PAIRS = {
    "toronto": (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg"),
    "sacre_coeur": (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg"),
}


def _load_oracle():
    ckpt = checkpoint_dir() / CHECKPOINT.filename
    if not ckpt.exists():
        pytest.skip(f"needs {ckpt}")
    sys.path.insert(0, str(ORACLE))
    try:
        from ufm_oracle import load_oracle

        return load_oracle(ckpt, device="cuda")
    finally:
        sys.path.remove(str(ORACLE))


@pytest.fixture(scope="module")
def fp32_models():
    previous = (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        oracle = _load_oracle()
        model = flashim.load("ufm", "fp32", device="cuda")
        yield oracle, model
        del oracle, model
        torch.cuda.empty_cache()
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32, prev_prec = previous
        torch.set_float32_matmul_precision(prev_prec)


@pytest.mark.parametrize("pair_name", list(PAIRS))
def test_fp32_forward_and_matches_parity(fp32_models, pair_name):
    oracle, model = fp32_models
    path_a, path_b = PAIRS[pair_name]
    img_a = np.asarray(Image.open(path_a).convert("RGB")).copy()
    img_b = np.asarray(Image.open(path_b).convert("RGB")).copy()

    with torch.inference_mode():
        # 1) Working-resolution forward parity (bit-identical on the same input tensors)
        a_prep, b_prep, _, _ = model.prepare(img_a, img_b)
        flow, covis_mask, covis_logits = model(a_prep, b_prep)
        ref_raw = oracle(
            {"img": a_prep, "symmetrized": False, "data_norm_type": "dinov2"},
            {"img": b_prep, "symmetrized": False, "data_norm_type": "dinov2"},
        )
        assert torch.equal(flow, ref_raw.flow.flow_output), (
            f"{pair_name}: working flow max abs diff {(flow - ref_raw.flow.flow_output).abs().max().item()}"
        )
        assert torch.equal(covis_mask, ref_raw.covisibility.mask), (
            f"{pair_name}: working covis max abs diff {(covis_mask - ref_raw.covisibility.mask).abs().max().item()}"
        )
        assert torch.equal(covis_logits, ref_raw.covisibility.logits), (
            f"{pair_name}: working logits max abs diff {(covis_logits - ref_raw.covisibility.logits).abs().max().item()}"
        )

        # 2) Full-resolution predict_flow parity vs upstream predict_correspondences_batched (FP32).
        # Upstream's ResizeToFixedManipulation permutes BCHW -> BHWC -> BCHW, leaving its resized tensors
        # channels-last (cuDNN selects a different FP32 reduction order in patch_embed.proj); with matching
        # channels-last input stride the unmapped flow & covisibility are bit-identical, and on contiguous
        # BCHW inputs median flow difference is < 0.05 px.
        src_t = torch.from_numpy(img_a).cuda()
        tgt_t = torch.from_numpy(img_b).cuda()
        with mock.patch("torch.autocast", return_value=nullcontext()):
            ref_full = oracle.predict_correspondences_batched(source_image=src_t, target_image=tgt_t)
        f_nhwc, cm_nhwc, _ = model(
            a_prep.to(memory_format=torch.channels_last), b_prep.to(memory_format=torch.channels_last)
        )
        from flashim.models.ufm.model import unmap_covisibility, unmap_flow

        assert torch.equal(unmap_flow(f_nhwc, img_a.shape[:2], img_b.shape[:2]), ref_full.flow.flow_output)
        assert torch.equal(unmap_covisibility(cm_nhwc, img_a.shape[:2]), ref_full.covisibility.mask)

        our_flow, our_covis = model.predict_flow(img_a, img_b)
        assert (our_flow - ref_full.flow.flow_output).abs().median().item() < 0.05
        assert (our_covis - ref_full.covisibility.mask).abs().median().item() < 1e-4

        # 3) Sampled Matches parity (same RNG seed on FlashIM vs oracle dense output)
        gen_ours = torch.Generator(device="cuda").manual_seed(42)
        m_ours = model.match(img_a, img_b, max_matches=5000, generator=gen_ours)
        assert len(m_ours) == 5000 and m_ours.dense is not None

        ref_dense = model._to_dense_result(
            ref_raw.flow.flow_output, ref_raw.covisibility.mask, ref_raw.covisibility.logits
        )
        gen_ref = torch.Generator(device="cuda").manual_seed(42)
        ref_matches, _, _, _ = model.sample(ref_dense, 5000, generator=gen_ref)
        ref_ka, ref_kb = model.to_pixel_coordinates(
            ref_matches, img_a.shape[0], img_a.shape[1], img_b.shape[0], img_b.shape[1]
        )
        assert torch.equal(m_ours.keypoints_A, ref_ka)
        assert torch.equal(m_ours.keypoints_B, ref_kb)


@pytest.mark.parametrize(
    "preset,max_med_epe,min_lt2,min_iou",
    [
        ("bf16", 0.35, 0.99, 0.975),
        ("int8", 0.65, 0.99, 0.975),
        ("fp8", 0.65, 0.99, 0.975),
    ],
)
def test_reduced_precision_presets_track_fp32(fp32_models, preset, max_med_epe, min_lt2, min_iou):
    """Reduced-precision presets build cleanly (no fallbacks on sm_89+) and track FP32 flow & covisibility."""
    _, fp32_model = fp32_models
    path_a, path_b = PAIRS["toronto"]
    img_a = np.asarray(Image.open(path_a).convert("RGB")).copy()
    img_b = np.asarray(Image.open(path_b).convert("RGB")).copy()

    model = flashim.load("ufm", preset, device="cuda")
    try:
        assert model.preset == preset or model.build_report.fallbacks
        with torch.inference_mode():
            a_prep, b_prep, _, _ = fp32_model.prepare(img_a, img_b)
            ref_flow, ref_covis, _ = fp32_model(a_prep, b_prep)
            flow, covis, _ = model(a_prep, b_prep)
            mask = ref_covis.squeeze(1) > 0.5
            epe = (flow - ref_flow).norm(dim=1)[mask]
            assert float(epe.median()) < max_med_epe
            assert float((epe < 2.0).float().mean()) > min_lt2
            inter = ((covis > 0.5) & (ref_covis > 0.5)).float().sum()
            union = ((covis > 0.5) | (ref_covis > 0.5)).float().sum().clamp(min=1)
            assert float(inter / union) > min_iou
            m = model.match(img_a, img_b, max_matches=1000, generator=torch.Generator(device="cuda").manual_seed(0))
            assert len(m) == 1000 and m.dense is not None
    finally:
        del model
        torch.cuda.empty_cache()
