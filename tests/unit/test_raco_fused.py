"""RaCo's fused ranker ops (residual block, 5x5 reflect head): every impl against the reference and the torch modules
they replace."""

import pytest
import torch

from flashim import ops
from flashim.models.raco import RaCo, RaCoConfig
from flashim.models.raco.fuse import fold_batchnorm

pytestmark = pytest.mark.cuda


def _folded_raco(seed=0):
    torch.manual_seed(seed)
    model = RaCo(RaCoConfig(num_keypoints=64)).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_var.uniform_(0.5, 1.5)
            module.running_mean.normal_(std=0.3)
            module.weight.data.uniform_(0.5, 1.5)
    fold_batchnorm(model)
    return model.cuda()


def _rel_rms(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _bind(block, impl):
    bound = ops.raco_resblock.bind(
        block.conv1.weight, block.conv1.bias, block.conv2.weight, block.conv2.bias,
        block.match_dims.weight, block.match_dims.bias, impl=impl,
    )
    if impl not in (None, "reference", *bound.available):
        pytest.skip(f"{impl} impl unavailable here")
    return bound


def _nhwc(x, width):
    out = torch.zeros(x.shape[2], x.shape[3], width, device="cuda", dtype=torch.bfloat16)
    out[..., : x.shape[1]] = x[0].permute(1, 2, 0)
    return out


@pytest.mark.parametrize("impl", ["reference", "triton", "cuda"])
@pytest.mark.parametrize("h,w", [(64, 64), (75, 101), (200, 232)])  # tile-aligned and ragged edges
@pytest.mark.parametrize("first", [False, True])  # 12 -> 12 on a 16-wide buffer / 3 -> 12 on a 4-wide buffer
def test_resblock_matches_torch_module(impl, h, w, first):
    block = _folded_raco().ranker_head[0 if first else 1]
    bound = _bind(block, impl)
    cin, in_ch = (3, 4) if first else (12, 16)
    x = torch.randn(1, cin, h, w, device="cuda")
    ops.reset_call_counts()
    with torch.inference_mode():
        ref = block(x)
        out = bound(_nhwc(x, in_ch), cin)
    assert ops.call_counts()["raco_resblock"] == {impl: 1}
    assert _rel_rms(out[..., :12].permute(2, 0, 1)[None], ref) < 0.01
    assert out[..., 12:].abs().max() == 0  # padding channels stay exactly zero


def test_cuda_and_triton_resblocks_agree():
    block = _folded_raco().ranker_head[1]
    triton, cuda = _bind(block, "triton"), _bind(block, "cuda")
    x = torch.randn(130, 141, 16, device="cuda").to(torch.bfloat16)
    x[..., 12:] = 0
    assert _rel_rms(triton(x, 12), cuda(x, 12)) < 1e-3


def test_auto_prefers_cuda_then_triton():
    bound = _bind(_folded_raco().ranker_head[1], None)
    assert bound.available[-1] == "triton"
    assert bound.primary == ("cuda" if "cuda" in bound.available else "triton")


def test_loader_falls_back_without_a_toolchain(monkeypatch):
    from flashim.backends import cuda as _cuda

    monkeypatch.setattr(_cuda, "nvcc_path", lambda: None)
    _cuda.load.cache_clear()
    try:
        assert _cuda.load("raco_resblock") is None
    finally:
        _cuda.load.cache_clear()


@pytest.mark.parametrize("impl", ["reference", "triton"])
def test_head_matches_reflect_conv(impl):
    head = _folded_raco().ranker_head[-1]
    bound = ops.raco_head5x5.bind(head.weight, head.bias, impl=impl)
    x = torch.randn(1, 12, 70, 90, device="cuda").to(torch.bfloat16)
    with torch.inference_mode():
        ref = head(x.float())
        out = bound(_nhwc(x, 16))
    assert out.shape == ref.shape and _rel_rms(out, ref) < 1e-3


@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_fused_ranker_map_tracks_the_torch_ranker_and_keeps_the_keypoints(backend):
    model = _folded_raco(1)
    if backend not in _bind(model.ranker_head[1], None).available:
        pytest.skip(f"{backend} impl unavailable here")
    image = torch.rand(1, 3, 192, 224, device="cuda")
    x = model.normalize(image)
    with torch.inference_mode():
        ref = model.ranker_map(x)
        keypoints_ref, _ = model.extract_keypoints(image)
        model.set_compute_dtype(torch.bfloat16)
        model.use_fused_ranker(backend)
        assert model.ranker_backend == backend
        fused = model.ranker_map(x)
        keypoints, _ = model.extract_keypoints(image)
    assert _rel_rms(fused, ref) < 0.05
    a, b = (set(map(tuple, k[0].round().int().tolist())) for k in (keypoints_ref, keypoints))
    assert len(a & b) / len(a) > 0.8  # random weights make near-ties common; the real checkpoint keeps 99.8%


def test_use_fused_ranker_requires_folded_batchnorm():
    model = RaCo(RaCoConfig(num_keypoints=8)).eval().cuda()
    with pytest.raises(RuntimeError, match="BatchNorm"):
        model.use_fused_ranker()


@pytest.mark.parametrize("shape", [(1, 8, 64, 64), (2, 8, 75, 101), (1, 8, 1, 1), (1, 8, 256, 320)])
def test_score_head_cuda_matches_reference(shape):
    from flashim.backends.cuda import raco as cuda_raco

    if cuda_raco._score_head_library() is None:
        pytest.skip("nvcc unavailable")
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    w1 = torch.randn(4, 8, 3, 3, device="cuda") * 0.2
    w2 = torch.randn(4, 4, 3, 3, device="cuda") * 0.2
    w3 = torch.randn(1, 4, 3, 3, device="cuda") * 0.2
    bound = ops.raco_score_head.bind(w1, w2, w3).warmup()
    assert bound.primary == "cuda"
    ops.reset_call_counts()
    with torch.backends.cudnn.flags(allow_tf32=False):
        ref = ops.raco_score_head.get("reference")(x, w1, w2, w3)
        out = bound(x)
    assert ops.call_counts()["raco_score_head"] == {"reference": 1, "cuda": 1}
    assert out.shape == ref.shape == (shape[0], 1, shape[2], shape[3]) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_fused_detector_tracks_unfused_detector():
    model = _folded_raco(2)
    image = torch.rand(1, 3, 192, 224, device="cuda")
    x = model.normalize(image)
    with torch.inference_mode():
        ref_fp32 = model.detector_logits(x)
        enabled_fp32 = model.use_fused_detector()
        assert any(s.startswith("score_head:") for s in enabled_fp32)
        fused_fp32 = model.detector_logits(x)
        torch.testing.assert_close(fused_fp32, ref_fp32, atol=2e-4, rtol=2e-4)  # TF32 in unfused cuDNN vs exact FP32

        model._fused_score_head = None
        model.set_compute_dtype(torch.bfloat16)
        ref_bf16 = model.detector_logits(x)
        enabled_bf16 = model.use_fused_detector()
        assert any(s.startswith("block1:") for s in enabled_bf16)
        fused_bf16 = model.detector_logits(x)
    assert _rel_rms(fused_bf16, ref_bf16) < 0.03
