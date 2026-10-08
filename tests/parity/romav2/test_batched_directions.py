"""fp8 refines both match directions as one batch with one projection per feature map.
That must be bit-identical to refining each direction separately with the same kernels."""

import pytest
import torch

from ...conftest import ASSETS

pytestmark = [pytest.mark.cuda, pytest.mark.weights, pytest.mark.slow]


def test_stacked_directions_bit_exact(fp32_highest):
    from flashim.models.romav2.build import load_romav2

    model = load_romav2("fp8", setting="mega1500", engine="eager")
    assert model.batched_directions
    inputs = model.prepare(ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")
    with torch.inference_mode():
        stacked = model(*inputs)
        model.batched_directions = False
        separate = model(*inputs)
    for key in ("warp_AB", "confidence_AB", "warp_BA", "confidence_BA"):
        assert torch.equal(stacked[key], separate[key]), key
    for stride in (4, 2, 1):
        for direction in ("AB", "BA"):
            for field in ("warp", "confidence"):
                a, b = stacked[f"refiner_{stride}_{direction}"], separate[f"refiner_{stride}_{direction}"]
                assert torch.equal(a[field], b[field]), (stride, direction, field)
