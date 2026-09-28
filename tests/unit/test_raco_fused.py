"""RaCo's fused ranker kernels (residual block, 5x5 reflect head) against the torch modules they replace."""

import pytest
import torch

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


def _backend(name):
    from flashim.backends.cuda import raco as cuda_kernel
    from flashim.backends.triton import raco as triton_kernel

    if name == "cuda":
        if not cuda_kernel.available():
            pytest.skip("needs nvcc")
        return cuda_kernel.resblock, cuda_kernel.pack_block
    return triton_kernel.resblock, lambda packed: packed


@pytest.mark.parametrize("backend", ["triton", "cuda"])
@pytest.mark.parametrize("h,w", [(64, 64), (75, 101), (200, 232)])  # tile-aligned and ragged edges
@pytest.mark.parametrize("first", [False, True])  # 12 -> 12 on a 16-wide buffer / 3 -> 12 on a 4-wide buffer
def test_fused_resblock_matches_torch(backend, h, w, first):
    from flashim.backends.triton import raco as fused

    run, adapt = _backend(backend)
    model = _folded_raco()
    block = model.ranker_head[0 if first else 1]
    cin, in_ch = (3, 4) if first else (12, 16)
    x = torch.randn(1, cin, h, w, device="cuda")
    with torch.inference_mode():
        ref = block(x)
        pack = fused.pack_block(block.conv1.weight, block.conv2.weight, block.match_dims.weight)
        packed = adapt((pack[0], fused.pad_bias(block.conv1.bias), pack[1], fused.pad_bias(block.conv2.bias), pack[2], fused.pad_bias(block.match_dims.bias)))
        nhwc = torch.zeros(h, w, in_ch, device="cuda", dtype=torch.bfloat16)
        nhwc[..., :cin] = x[0].permute(1, 2, 0)
        out, _ = run(nhwc, cin, packed)
    assert _rel_rms(out[..., :12].permute(2, 0, 1)[None], ref) < 0.01
    assert out[..., 12:].abs().max() == 0  # padding channels stay exactly zero


def test_cuda_and_triton_resblocks_agree():
    from flashim.backends.cuda import raco as cuda_kernel
    from flashim.backends.triton import raco as fused

    if not cuda_kernel.available():
        pytest.skip("needs nvcc")
    block = _folded_raco().ranker_head[1]
    pack = fused.pack_block(block.conv1.weight, block.conv2.weight, block.match_dims.weight)
    packed = (pack[0], fused.pad_bias(block.conv1.bias), pack[1], fused.pad_bias(block.conv2.bias), pack[2], fused.pad_bias(block.match_dims.bias))
    x = torch.randn(130, 141, 16, device="cuda").to(torch.bfloat16)
    x[..., 12:] = 0
    a, _ = fused.resblock(x, 12, packed)
    b, _ = cuda_kernel.resblock(x, 12, cuda_kernel.pack_block(packed))
    assert _rel_rms(a, b) < 1e-3


def test_loader_falls_back_without_a_toolchain(monkeypatch):
    from flashim.backends import cuda as _cuda

    monkeypatch.setattr(_cuda, "nvcc_path", lambda: None)
    _cuda.load.cache_clear()
    try:
        assert _cuda.load("raco_resblock") is None
    finally:
        _cuda.load.cache_clear()


def test_fused_head_matches_reflect_conv():
    from flashim.backends.triton import raco as fused

    model = _folded_raco()
    head = model.ranker_head[-1]
    x = torch.randn(1, 12, 70, 90, device="cuda").to(torch.bfloat16)
    nhwc = torch.zeros(70, 90, 16, device="cuda", dtype=torch.bfloat16)
    nhwc[..., :12] = x[0].permute(1, 2, 0)
    with torch.inference_mode():
        ref = head(x.float())
        out = fused.head5x5(nhwc, fused.pack_head(head))
    assert out.shape == ref.shape and _rel_rms(out, ref) < 1e-3


@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_fused_ranker_map_tracks_the_torch_ranker_and_keeps_the_keypoints(backend):
    if backend == "cuda":
        _backend("cuda")
    model = _folded_raco(1)
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
