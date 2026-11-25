"""``ops.mutual_nn_match``: fused tiled Triton MNN matching against the torch reference."""

import pytest
import torch
import torch.nn.functional as F

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.mark.parametrize("shape", [(0, 5, 64), (5, 0, 64), (1, 1, 32), (17, 29, 64), (130, 260, 128), (1025, 777, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_mutual_nn_match_matches_reference(shape, dtype, threshold):
    torch.manual_seed(0)
    m, n, d = shape
    desc0 = F.normalize(torch.randn(m, d, device=DEV), dim=-1).to(dtype)
    desc1 = F.normalize(torch.randn(n, d, device=DEV), dim=-1).to(dtype)
    # Plant clear ground-truth mutual pairs so the match set is non-trivial even at high threshold.
    k = min(m, n, 20)
    if k > 0:
        desc1[:k] = desc0[:k]
    ops.reset_call_counts()
    ia, ib, sc = ops.mutual_nn_match(desc0, desc1, threshold)
    ref_ia, ref_ib, ref_sc = ops.mutual_nn_match.get("reference")(desc0, desc1, threshold)
    assert ops.call_counts()["mutual_nn_match"] == {"triton": 1, "reference": 1}
    assert ia.dtype == torch.int64 and ib.dtype == torch.int64 and sc.dtype == torch.float32
    if m == 0 or n == 0:
        assert len(ia) == len(ref_ia) == 0
        return
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    # BF16 rounding can flip rare borderline non-planted pairs; FP32 <= 8192^2 uses IEEE precision and matches exactly.
    ours = set(zip(ia.tolist(), ib.tolist()))
    theirs = set(zip(ref_ia.tolist(), ref_ib.tolist()))
    overlap = len(ours & theirs) / max(len(theirs), 1)
    min_overlap = 0.99 if dtype == torch.bfloat16 else 0.999
    assert overlap >= min_overlap, f"overlap={overlap:.4f} ({len(ours)} vs {len(theirs)})"
    common_scores = {
        (i, j): s for i, j, s in zip(ia.tolist(), ib.tolist(), sc.tolist())
    }
    ref_scores = {
        (i, j): s for i, j, s in zip(ref_ia.tolist(), ref_ib.tolist(), ref_sc.tolist())
    }
    max_diff = max((abs(common_scores[p] - ref_scores[p]) for p in ours & theirs), default=0.0)
    assert max_diff < tol
