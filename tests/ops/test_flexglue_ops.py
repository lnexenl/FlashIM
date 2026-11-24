"""CUDA op tests for FlexGlue kernels (entmax readout, 3D KDE, QKV/QK-norm attention glue)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops
from flashim.models.flexglue.model import cell_centers

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize(
    "b,n,hw,radius,scale",
    [
        (1, 5, (3, 4), 1, 1.0),
        (2, 17, (7, 9), 1, 5.0),
        (1, 33, (16, 16), 2, 10.0),
        (1, 64, (32, 32), 1, 10.0),
        (1, 128, (64, 64), 1, 10.0),
        (1, 2048, (64, 64), 1, 10.0),
        (1, 4096, (64, 64), 1, 10.0),
        (1, 16, (50, 75), 1, 8.0),  # non-power-of-2 > 512
    ],
)
@torch.inference_mode()
def test_flexglue_entmax_readout(b, n, hw, radius, scale):
    torch.manual_seed(0)
    h, w = hw
    logits = torch.randn(b, n, h * w, device="cuda", dtype=torch.float32) * scale
    grid = cell_centers(h, w, scale=(w, h), device=logits.device)

    ref_pos, ref_mass = ops.flexglue_entmax_readout.get("reference")(logits, grid, hw, radius=radius)
    tri_pos, tri_mass = ops.flexglue_entmax_readout.get("triton")(logits, grid, hw, radius=radius)

    assert ops.flexglue_entmax_readout.explain(logits, grid, hw, radius=radius)["triton"] == "selected"
    assert torch.allclose(tri_pos, ref_pos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tri_mass, ref_mass, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
def test_flexglue_entmax_readout_edge_cases():
    h, w = 64, 64
    grid = cell_centers(h, w, scale=(w, h), device="cuda")

    # 1. Border corners (argmax at (0,0), (0,w-1), (h-1,0), (h-1,w-1))
    logits = torch.full((1, 4, h * w), -10.0, device="cuda")
    for i, idx in enumerate((0, w - 1, (h - 1) * w, h * w - 1)):
        logits[0, i, idx] = 10.0
    ref_pos, ref_mass = ops.flexglue_entmax_readout.get("reference")(logits, grid, (h, w))
    tri_pos, tri_mass = ops.flexglue_entmax_readout.get("triton")(logits, grid, (h, w))
    assert torch.allclose(tri_pos, ref_pos, atol=1e-6)
    assert torch.allclose(tri_mass, ref_mass, atol=1e-6)

    # 2. Sharp peak (support = 1) and flat cluster (support = 100 equal logits)
    logits = torch.full((1, 2, h * w), -20.0, device="cuda")
    logits[0, 0, 1000] = 5.0
    logits[0, 1, 500:600] = 5.0
    ref_pos, ref_mass = ops.flexglue_entmax_readout.get("reference")(logits, grid, (h, w))
    tri_pos, tri_mass = ops.flexglue_entmax_readout.get("triton")(logits, grid, (h, w))
    assert torch.allclose(tri_pos, ref_pos, atol=1e-6)
    assert torch.allclose(tri_mass, ref_mass, atol=1e-6)


@torch.inference_mode()
def test_kde_density_batched():
    torch.manual_seed(0)
    x = torch.rand(2, 512, 4, device="cuda", dtype=torch.float32) * 2 - 1
    assert ops.kde_density.explain(x, std=0.1)["triton"] == "selected"
    batched = ops.kde_density.get("triton")(x, std=0.1)
    assert batched.shape == (2, 512)
    for i in range(2):
        single = ops.kde_density.get("triton")(x[i], std=0.1)
        assert torch.equal(batched[i], single)
