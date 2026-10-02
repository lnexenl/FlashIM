"""``ops.dual_softmax_mutual``: the streaming Triton impl against the torch reference."""

import pytest
import torch

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


def _both(*args):
    return ops.dual_softmax_mutual.get("triton")(*args), ops.dual_softmax_mutual.get("reference")(*args)


# tiny, odd, non-multiples of the 8x512 / 64x64 tiles, wide and tall, and a LightGlue-sized square
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (5, 1), (3, 5), (17, 513), (65, 63), (129, 1025), (700, 9), (2048, 2048)])
@pytest.mark.parametrize("scale", [1.0, 30.0])
@pytest.mark.parametrize("biased", [False, True])
def test_matches_reference(shape, scale, biased):
    torch.manual_seed(0)
    rows, cols = shape
    sim = torch.randn(rows, cols, device=DEV) * scale
    bias = (-torch.rand(rows, device=DEV), -torch.rand(cols, device=DEV)) if biased else (None, None)
    (log_conf, index, mutual), (ref_conf, ref_index, ref_mutual) = _both(sim, *bias)
    assert index.dtype == torch.int64 and mutual.dtype == torch.bool and log_conf.dtype == torch.float32
    # continuous random scores have no exact ties: the argmax indices must agree exactly
    assert torch.equal(index, ref_index)
    assert torch.equal(mutual, ref_mutual)
    # two log-sum-exps evaluated in a different order: a few ulp of values up to ``2 * scale * 5``
    assert (log_conf - ref_conf).abs().max() <= 1e-4 * max(1.0, scale)


def test_large_values_do_not_overflow():
    sim = torch.full((40, 50), 200.0, device=DEV)  # exp(200) overflows FP32 without the running-maximum rescale
    sim[3, 7] = 300.0
    (log_conf, index, mutual), (ref_conf, _, _) = _both(sim)
    assert torch.isfinite(log_conf).all()
    assert index[3] == 7 and mutual[3]
    assert (log_conf - ref_conf).abs().max() <= 1e-3


def test_dispatch_selects_triton_for_fp32_matrices():
    ops.reset_call_counts()
    sim = torch.randn(300, 200, device=DEV)
    ops.dual_softmax_mutual(sim)
    assert ops.call_counts()["dual_softmax_mutual"] == {"triton": 1}
    ops.dual_softmax_mutual(sim.double())  # not FP32: the reference
    assert ops.call_counts()["dual_softmax_mutual"] == {"triton": 1, "reference": 1}
