"""Fused multi-scale fusion (upsample + concat + score-head 1x1 conv + SELU + L2 normalize): reference parity, CUDA dispatch."""

import pytest
import torch
import torch.nn.functional as F

from flashim.ops.aliked import _multiscale_fuse_reference


def _manual(f1, f2, f3, f4, weight):
    size = f1.shape[-2:]
    fused = torch.cat([f1.float(), *(F.interpolate(f.float(), size=size, mode="bilinear", align_corners=True) for f in (f2, f3, f4))], dim=1)
    return F.selu(F.conv2d(fused, weight[:, :, None, None])), F.normalize(fused, p=2, dim=1).to(f1.dtype)


def _inputs(dtype, c, h, w, device="cpu"):
    f1 = torch.randn(1, c, h, w, dtype=dtype, device=device)
    f2 = torch.randn(1, c, h // 2, w // 2, dtype=dtype, device=device)
    f3 = torch.randn(1, c, max(1, h // 8), max(1, w // 8), dtype=dtype, device=device)
    f4 = torch.randn(1, c, max(1, h // 32), max(1, w // 32), dtype=dtype, device=device)
    weight = torch.randn(8, 4 * c, device=device) * 0.2
    return f1, f2, f3, f4, weight


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_matches_manual_chain(dtype):
    torch.manual_seed(0)
    f1, f2, f3, f4, weight = _inputs(dtype, 32, 96, 128)
    head, feature = _multiscale_fuse_reference(f1, f2, f3, f4, weight)
    ref_head, ref_feature = _manual(f1, f2, f3, f4, weight)
    assert torch.equal(head, ref_head) and torch.equal(feature, ref_feature)
    assert head.dtype == torch.float32 and feature.dtype == dtype


def test_reference_skips_unrequested_outputs():
    f1, f2, f3, f4, weight = _inputs(torch.float32, 16, 64, 64)
    head, feature = _multiscale_fuse_reference(f1, f2, f3, f4, None, normalize=True)
    assert head is None and feature.shape == (1, 64, 64, 64)
    head, feature = _multiscale_fuse_reference(f1, f2, f3, f4, weight, normalize=False)
    assert feature is None and head.shape == (1, 8, 64, 64)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("c,h,w", [(32, 100, 130), (16, 96, 96)])
@pytest.mark.parametrize("with_head,normalize", [(True, True), (True, False), (False, True)])
def test_triton_kernel_matches_reference_on_cuda(dtype, c, h, w, with_head, normalize):
    from flashim import ops

    torch.manual_seed(1)
    f1, f2, f3, f4, weight = _inputs(dtype, c, h, w, "cuda")
    weight = weight if with_head else None
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # the reference conv would otherwise round to TF32; the kernel is ~FP32
    try:
        ref_head, ref_feature = _multiscale_fuse_reference(f1, f2, f3, f4, weight, normalize=normalize)
    finally:
        torch.backends.cudnn.allow_tf32 = previous
    head, feature = ops.multiscale_fuse(f1, f2, f3, f4, weight, normalize=normalize)
    assert (head is None) == (ref_head is None) and (feature is None) == (ref_feature is None)
    if head is not None:
        assert (head - ref_head).abs().max() < 2e-4  # tf32x3 dot + coordinate rounding
    if feature is not None:
        cosine = F.cosine_similarity(feature.float(), ref_feature.float(), dim=1)
        assert cosine.min() > 0.9999 and (feature.float() - ref_feature.float()).abs().max() < 5e-3
