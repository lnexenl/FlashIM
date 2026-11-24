"""Exactness tests for ``ops.box_nms_2d`` (Triton iterative 2D box-NMS vs torchvision.ops.nms reference)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops


@pytest.mark.parametrize("shape", [(1, 1), (8, 8), (63, 79), (256, 320), (728, 1024)])
@pytest.mark.cuda
def test_box_nms_2d_triton_exact_match(shape: tuple[int, int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    g = torch.Generator(device="cuda").manual_seed(17 + shape[0] * 13 + shape[1])
    # Realistic sparse peak distribution (~2-5% of pixels >= 0.01) + deliberate exact ties
    scores = torch.rand(shape, device="cuda", dtype=torch.float32, generator=g).pow(16)
    if shape[0] >= 8 and shape[1] >= 8:
        scores[3, 3] = 0.75
        scores[3, 4] = 0.75  # exact tie within suppression zone
        scores[4, 4] = 0.50

    ref_kpts, ref_vals = ops.box_nms_2d.get("reference")(scores, 0.01, 4, 0.1)
    got_kpts, got_vals = ops.box_nms_2d(scores, 0.01, 4, 0.1)

    assert ops.box_nms_2d.explain(scores, 0.01, 4, 0.1)["triton"] == "selected"
    ref_set = {(int(x), int(y)) for x, y in ref_kpts.cpu().tolist()}
    got_set = {(int(x), int(y)) for x, y in got_kpts.cpu().tolist()}
    assert got_set == ref_set
    # Also verify exact score values per keypoint
    ref_map = {(int(x), int(y)): v for (x, y), v in zip(ref_kpts.cpu().tolist(), ref_vals.cpu().tolist())}
    got_map = {(int(x), int(y)): v for (x, y), v in zip(got_kpts.cpu().tolist(), got_vals.cpu().tolist())}
    assert got_map == ref_map


@pytest.mark.cuda
def test_box_nms_2d_empty_and_all_below_threshold() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    scores = torch.zeros((64, 64), device="cuda", dtype=torch.float32)
    kpts, vals = ops.box_nms_2d(scores, 0.01, 4, 0.1)
    assert kpts.shape == (0, 2)
    assert vals.shape == (0,)
