"""DeDoDe's ``low_precision_similarity`` pass (``dedode.spec``): ``match_features``' similarity on BF16 tensor cores."""

from types import SimpleNamespace

import torch

from flashim import ops
from flashim.models.dedode import spec
from flashim.models.dedode.model import DeDoDe


def _descriptors(m=64, n=48, d=32):
    torch.manual_seed(0)
    return (torch.nn.functional.normalize(torch.randn(m, d), dim=-1), torch.nn.functional.normalize(torch.randn(n, d), dim=-1))


def test_similarity_default_is_the_fp32_gemm():
    a, b = _descriptors()
    model = SimpleNamespace(low_precision_similarity=False, inv_temperature=20.0)
    assert not DeDoDe.low_precision_similarity
    assert torch.equal(DeDoDe.similarity(model, a, b), (a @ b.t()) * 20.0)


def test_low_precision_similarity_rounds_the_descriptors_to_bf16():
    a, b = _descriptors()
    model = SimpleNamespace(low_precision_similarity=True, inv_temperature=20.0)
    out = DeDoDe.similarity(model, a, b)
    assert out.dtype == torch.float32 and out.shape == (64, 48)
    want = ops.matmul_nt_fp32.get("reference")(a.bfloat16(), b.bfloat16(), 20.0)
    torch.testing.assert_close(out, want, atol=1e-4, rtol=1e-5)  # CPU: the reference itself, up to summation order
    assert (out - (a @ b.t()) * 20.0).abs().max() < 0.05  # BF16 rounding of unit-norm descriptors, logits in [-20, 20]


def test_pass_is_registered_and_skips_fp32():
    for model_spec in (spec.SPEC, spec.SPEC_G):
        assert "low_precision_similarity" in model_spec.passes
        assert not model_spec.passes["low_precision_similarity"].exact
        assert "low_precision_similarity" in model_spec.presets["fp32"].get("optimize", {}).get("passes", ())
    ctx = SimpleNamespace(config=SimpleNamespace(precision=SimpleNamespace(compute_dtype="fp32")))
    model = torch.nn.Module()
    assert spec.low_precision_similarity(model, ctx) == "skipped: fp32"
