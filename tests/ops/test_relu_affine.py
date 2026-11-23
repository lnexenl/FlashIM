"""Correctness tests for ``ops.relu_affine``."""

from __future__ import annotations

import pytest
import torch

from flashim import ops


@pytest.mark.parametrize("c", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.cuda
def test_relu_affine_triton_matches_reference(c: int, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    g = torch.Generator(device="cuda").manual_seed(7 + c)
    x = torch.randn((1, c, 91, 128), device="cuda", dtype=dtype, generator=g).to(memory_format=torch.channels_last)
    scale = torch.randn(c, device="cuda", dtype=torch.float32, generator=g) * 0.5 + 1.0
    shift = torch.randn(c, device="cuda", dtype=torch.float32, generator=g) * 0.2

    ref = ops.relu_affine.get("reference")(x, scale, shift)
    got = ops.relu_affine(x, scale, shift)
    assert ops.relu_affine.explain(x, scale, shift)["triton"] == "selected"
    atol = 1e-6 if dtype == torch.float32 else 2e-3
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
