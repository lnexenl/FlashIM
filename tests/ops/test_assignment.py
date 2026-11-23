"""Tests for :mod:`flashim.ops.assignment` (``log_optimal_transport``)."""

import pytest
import torch

from flashim import ops


def test_log_optimal_transport_marginals_and_shape():
    torch.manual_seed(0)
    scores = torch.randn(2, 16, 20)
    alpha = torch.tensor(1.0)
    log_p = ops.log_optimal_transport(scores, alpha, 100)
    assert log_p.shape == (2, 17, 21)
    p = log_p.exp()
    # Non-dustbin rows and columns sum to 1; dustbin column sums to N, dustbin row sums to M.
    assert torch.allclose(p[:, :-1, :].sum(dim=2), torch.ones(2, 16), atol=1e-4)
    assert torch.allclose(p[:, :, :-1].sum(dim=1), torch.ones(2, 20), atol=1e-4)
    assert torch.allclose(p[:, -1, :].sum(dim=1), torch.full((2,), 20.0), atol=1e-3)
    assert torch.allclose(p[:, :, -1].sum(dim=1), torch.full((2,), 16.0), atol=1e-3)


def test_log_optimal_transport_strong_diagonal_matches():
    m = 8
    scores = torch.eye(m)[None] * 20.0
    log_p = ops.log_optimal_transport(scores, torch.tensor(0.0), 50)
    diag = log_p[0, :m, :m].diagonal().exp()
    assert (diag > 0.99).all()


def test_log_optimal_transport_high_dustbin_rejects():
    scores = torch.zeros(1, 5, 6)
    log_p = ops.log_optimal_transport(scores, torch.tensor(20.0), 100)
    assert (log_p[0, :-1, :-1].exp() < 1e-3).all()
    assert (log_p[0, :-1, -1].exp() > 0.99).all()


def test_log_optimal_transport_validation():
    with pytest.raises(ValueError, match="shape"):
        ops.log_optimal_transport(torch.zeros(4, 4), torch.tensor(1.0), 10)
    with pytest.raises(ValueError, match="non-negative"):
        ops.log_optimal_transport(torch.zeros(1, 4, 4), torch.tensor(1.0), -1)
    with pytest.raises(ValueError, match="non-empty"):
        ops.log_optimal_transport(torch.zeros(1, 0, 4), torch.tensor(1.0), 10)


@pytest.mark.cuda
def test_log_optimal_transport_cuda_matches_cpu():
    torch.manual_seed(1)
    scores = torch.randn(1, 32, 24)
    alpha = torch.tensor(1.5)
    cpu = ops.log_optimal_transport(scores, alpha, 100)
    gpu = ops.log_optimal_transport(scores.cuda(), alpha.cuda(), 100).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-4)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("b", "m", "n", "iters"),
    [
        (1, 1, 1, 10),
        (1, 1, 17, 20),
        (1, 17, 1, 20),
        (1, 37, 53, 100),
        (1, 64, 128, 0),
        (2, 129, 257, 100),
        (1, 816, 1024, 100),
        (1, 1024, 1024, 100),
        (1, 1500, 1200, 100),
    ],
)
def test_log_optimal_transport_triton_matches_reference(b, m, n, iters):
    torch.manual_seed(42 + m + n)
    scores = torch.randn(b, m, n, device="cuda", dtype=torch.float32) * 2.0
    alpha = torch.tensor(1.3, device="cuda", dtype=torch.float32)
    explain = ops.log_optimal_transport.explain(scores, alpha, iters)
    assert any(v == "selected" for v in explain.values()), explain
    ref = ops.log_optimal_transport.get("reference")(scores, alpha, iters)
    for impl in ("triton", "cuda"):
        if explain.get(impl) not in ("selected", "lower priority"):
            continue
        got = ops.log_optimal_transport.get(impl)(scores, alpha, iters)
        assert got.shape == ref.shape
        # Base-2 exp2/log2 vs base-e exp/log over 100 iterations differs by ~1e-5 in log space, ~1e-6 in prob space.
        # Note: the (M, N) dustbin corner has exp() ~ max(M, N) (not a probability in [0, 1]), so its relative
        # error is bounded by the log-space check; check probabilities on all [0, 1] match and dustbin entries.
        assert float((got - ref).abs().max()) < 5e-4, f"{impl}: max log diff"
        assert float((got[:, :-1, :].exp() - ref[:, :-1, :].exp()).abs().max()) < 5e-5, f"{impl}: max prob diff"
        assert float((got[:, -1, :-1].exp() - ref[:, -1, :-1].exp()).abs().max()) < 5e-5, f"{impl}: max dustbin diff"
