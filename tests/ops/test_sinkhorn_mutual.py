"""Correctness and CUDA-graph tests for ``ops.sinkhorn_mutual``."""

from __future__ import annotations

import pytest
import torch

from flashim import ops


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (64, 64), (513, 777), (1024, 1024), (1200, 1100)])
@pytest.mark.parametrize("iters", [0, 10, 100])
@pytest.mark.cuda
def test_sinkhorn_mutual_cuda_matches_reference(shape: tuple[int, int], iters: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    g = torch.Generator(device="cuda").manual_seed(42 + shape[0] * 7 + shape[1] + iters)
    sim = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g) * 2.5
    bin_score = torch.tensor(1.37, device="cuda", dtype=torch.float32)

    ref_prob, ref_idx, ref_mut = ops.sinkhorn_mutual.get("reference")(sim, bin_score, iters)
    got_prob, got_idx, got_mut = ops.sinkhorn_mutual(sim, bin_score, iters)

    assert ops.sinkhorn_mutual.explain(sim, bin_score, iters)["cuda"] == "selected"
    torch.testing.assert_close(got_prob, ref_prob, atol=1e-5, rtol=1e-4)
    assert torch.equal(got_idx, ref_idx)
    assert torch.equal(got_mut, ref_mut)


@pytest.mark.cuda
def test_sinkhorn_mutual_cuda_graph_capture() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from flashim.engines.cuda_graph import CudaGraphRunner

    g = torch.Generator(device="cuda").manual_seed(123)
    sim = torch.randn((512, 512), device="cuda", dtype=torch.float32, generator=g) * 2.0
    bin_score = torch.tensor(1.2, device="cuda", dtype=torch.float32)

    runner = CudaGraphRunner(lambda s, b: ops.sinkhorn_mutual(s, b, 100))
    p1, i1, m1 = runner(sim, bin_score)
    ref_p, ref_i, ref_m = ops.sinkhorn_mutual.get("reference")(sim, bin_score, 100)
    torch.testing.assert_close(p1, ref_p, atol=1e-5, rtol=1e-4)
    assert torch.equal(i1, ref_i)
    assert torch.equal(m1, ref_m)

    # Replay with different values
    sim2 = torch.randn((512, 512), device="cuda", dtype=torch.float32, generator=g) * 2.0
    p2, i2, m2 = runner(sim2, bin_score)
    ref_p2, ref_i2, ref_m2 = ops.sinkhorn_mutual.get("reference")(sim2, bin_score, 100)
    torch.testing.assert_close(p2, ref_p2, atol=1e-5, rtol=1e-4)
    assert torch.equal(i2, ref_i2)
    assert torch.equal(m2, ref_m2)
