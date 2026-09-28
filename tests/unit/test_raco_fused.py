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
