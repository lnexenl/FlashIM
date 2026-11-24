"""Correctness tests for descriptor matching ops (``hamming_distances``, ``hamming_ratio_mutual_match``,
``ratio_mutual_match``): bit-identical to the torch references across odd/tiny/real sizes and tie cases."""

from __future__ import annotations

import pytest
import torch

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.mark.parametrize("n,m,d", [(0, 10, 32), (10, 0, 32), (1, 2, 32), (7, 13, 32), (65, 129, 32), (333, 517, 16), (1024, 1024, 32)])
def test_hamming_distances_bit_identical(n, m, d):
    torch.manual_seed(0)
    a = torch.randint(0, 256, (n, d), device=DEV, dtype=torch.uint8)
    b = torch.randint(0, 256, (m, d), device=DEV, dtype=torch.uint8)
    assert ops.hamming_distances.explain(a, b)["triton"] == "selected"
    fast = ops.hamming_distances.get("triton")(a, b)
    ref = ops.hamming_distances.get("reference")(a, b)
    assert torch.equal(fast, ref)


@pytest.mark.parametrize("n,m,ratio", [(0, 10, 0.8), (10, 1, 0.8), (1, 2, 0.8), (33, 67, 0.8), (513, 499, 0.75), (2048, 2048, 0.8)])
def test_hamming_ratio_mutual_match_bit_identical(n, m, ratio):
    torch.manual_seed(1)
    a = torch.randint(0, 256, (n, 32), device=DEV, dtype=torch.uint8)
    b = torch.randint(0, 256, (m, 32), device=DEV, dtype=torch.uint8)
    # Plant exact and near matches plus duplicate rows/cols to stress tie-breaking.
    if n >= 10 and m >= 10:
        b[:5] = a[:5]
        b[5] = a[0]  # tie in row 0 -> must be rejected by ratio test
        a[6] = a[1]  # tie in col 1 -> smallest row index (1) wins col argmin
    assert ops.hamming_ratio_mutual_match.explain(a, b, ratio)["triton"] == "selected"
    ia_f, ib_f, c_f = ops.hamming_ratio_mutual_match.get("triton")(a, b, ratio)
    ia_r, ib_r, c_r = ops.hamming_ratio_mutual_match.get("reference")(a, b, ratio)
    assert torch.equal(ia_f, ia_r)
    assert torch.equal(ib_f, ib_r)
    assert torch.equal(c_f, c_r)


@pytest.mark.parametrize("n,m,ratio", [(0, 10, 0.8), (10, 1, 0.8), (1, 2, 0.8), (37, 71, 0.8), (513, 499, 0.75), (2048, 2048, 0.8)])
def test_ratio_mutual_match_bit_identical(n, m, ratio):
    torch.manual_seed(2)
    a = torch.nn.functional.normalize(torch.randn(n, 128, device=DEV), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(m, 128, device=DEV), dim=-1)
    if n >= 10 and m >= 10:
        b[:5] = a[:5]
        b[5] = a[0]
        a[6] = a[1]
    dist = torch.cdist(a, b)
    assert ops.ratio_mutual_match.explain(dist, ratio)["triton"] == "selected"
    ia_f, ib_f, c_f = ops.ratio_mutual_match.get("triton")(dist, ratio)
    ia_r, ib_r, c_r = ops.ratio_mutual_match.get("reference")(dist, ratio)
    assert torch.equal(ia_f, ia_r)
    assert torch.equal(ib_f, ib_r)
    assert torch.equal(c_f, c_r)
