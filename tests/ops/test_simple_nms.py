"""Correctness tests for ``ops.simple_nms`` (CUDA C++ shared-memory separable NMS vs torch reference)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.mark.parametrize("radius", [2, 4])
@pytest.mark.parametrize("shape", [(1, 1, 7, 9), (1, 1, 37, 45), (1, 856, 1200), (2, 1, 130, 173)])
def test_simple_nms_cuda_bit_identical(radius, shape):
    torch.manual_seed(0)
    scores = torch.sigmoid(torch.randn(shape, device=DEV, dtype=torch.float32) * 2.0)
    # Plant plateaus / ties to stress exact equality semantics across both NMS rounds.
    if shape[-2] >= 12 and shape[-1] >= 12:
        scores[..., 4:9, 4:9] = 0.99
    assert ops.simple_nms.explain(scores, radius)["cuda"] == "selected"
    fast = ops.simple_nms.get("cuda")(scores, radius)
    ref = ops.simple_nms.get("reference")(scores, radius)
    assert torch.equal(fast, ref)
