"""Quantized deformable conv: gather-math parity vs torchvision, calibration, and end-to-end accuracy.

CPU-only tests use the reference (pure-PyTorch) op impl; ``@pytest.mark.cuda`` tests check the Triton
kernel against that same reference.
"""

import platform

import pytest
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from flashim.models.aliked import ALIKED, ALIKEDConfig
from flashim.models.aliked.deform_quant import (
    DeformConvCalibrator,
    QuantizedDeformableConv2d,
    quantize_deformable_convs,
)
from flashim.models.aliked.fuse import fold_batchnorm
from flashim.ops.aliked import _deform_patches_int8_reference, gather_deform_patches
from flashim.quant.calibration import Int8CalibrationTable

SMALL = ALIKEDConfig(c1=4, c2=8, c3=8, c4=8, dim=8, K=3, M=4)


@pytest.mark.parametrize("offset_scale", [2.0, 40.0])  # 40.0 pushes samples far outside the image
def test_gather_matches_torchvision_deform_conv2d(offset_scale):
    torch.manual_seed(0)
    b, c, h, w, k = 2, 8, 17, 23, 3
    x = torch.randn(b, c, h, w)
    offset = (torch.rand(b, 2 * k * k, h, w) - 0.5) * offset_scale
    weight = torch.randn(12, c, k, k)
    bias = torch.randn(12)

    reference = deform_conv2d(x, offset, weight, bias=bias, padding=(1, 1))
    patches = gather_deform_patches(x, offset, k, padding=1)  # [B, C, K*K, oH, oW]
    ours = torch.einsum("oc,bcp->bop", weight.reshape(12, -1), patches.reshape(b, -1, h * w)).reshape(b, 12, h, w)
    ours = ours + bias.view(1, -1, 1, 1)
    if platform.machine() == "x86_64":
        assert torch.equal(reference, ours)  # bit-exact on CPU
    else:  # aarch64 (Jetson Orin, OpenBLAS): up to 2 ulp (3.8e-6), both sides equally far from an FP64 GEMM
        torch.testing.assert_close(ours, reference, rtol=0, atol=2e-5)


def test_gather_zero_offset_equals_plain_unfold():
    torch.manual_seed(1)
    b, c, h, w, k = 1, 4, 9, 11, 3
    x = torch.randn(b, c, h, w)
    offset = torch.zeros(b, 2 * k * k, h, w)
    patches = gather_deform_patches(x, offset, k, padding=1)  # [B, C, K*K, oH, oW]
    unfolded = F.unfold(x, kernel_size=k, padding=1).view(b, c, k * k, h, w)
    assert torch.equal(patches, unfolded)


def test_quantize_int8_static_roundtrips_within_one_quantum():
    torch.manual_seed(2)
    x = torch.randn(1, 8, 6, 6)
    offset = (torch.rand(1, 18, 6, 6) - 0.5) * 3.0
    scale = torch.tensor(0.02)
    quantized = _deform_patches_int8_reference(x, offset, 3, 1, scale)
    patches = gather_deform_patches(x, offset, 3, 1).permute(0, 3, 4, 1, 2).reshape(1, 6, 6, -1)
    assert (quantized.dequantize() - patches).abs().max() <= scale  # +-0.5 quantum, safely bounded by 1


def test_calibrator_and_table_roundtrip(tmp_path):
    torch.manual_seed(3)
    model = ALIKED(SMALL).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_var.uniform_(0.5, 1.5)
    fold_batchnorm(model)
    names = ["block3.conv1", "block3.conv2", "block4.conv1", "block4.conv2"]
    with torch.inference_mode(), DeformConvCalibrator(model, names) as calibrator:
        model(torch.rand(2, 3, 64, 80))
    table = calibrator.table()
    assert set(table.input_scale) == set(names)
    assert all(float(table[name]) > 0 for name in names)

    path = tmp_path / "aliked_int8.safetensors"
    table.save(path)
    reloaded = Int8CalibrationTable.load(path)
    for name in names:
        assert torch.equal(table[name], reloaded[name])


def test_calibrator_rejects_non_deformable_module():
    model = ALIKED(SMALL).eval()
    with pytest.raises(TypeError):
        DeformConvCalibrator(model, ["block1"])  # block1 is a plain ConvBlock, no DeformableConv2d


def test_quantized_deform_conv_tracks_fp32_within_noise():
    torch.manual_seed(4)
    model = ALIKED(SMALL).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_var.uniform_(0.5, 1.5)
    fold_batchnorm(model)
    names = ["block3.conv1", "block3.conv2", "block4.conv1", "block4.conv2"]
    calib_image = torch.rand(2, 3, 64, 80)
    with torch.inference_mode(), DeformConvCalibrator(model, names) as calibrator:
        model(calib_image)
    table = calibrator.table()

    test_image = torch.rand(1, 3, 64, 80)
    with torch.inference_mode():
        feature_before, score_before = model(test_image)

    replaced = quantize_deformable_convs(model, table)
    assert replaced == names
    assert all(isinstance(dict(model.named_modules())[n], QuantizedDeformableConv2d) for n in names)

    with torch.inference_mode():
        feature_after, score_after = model(test_image)
    cosine = F.cosine_similarity(feature_before, feature_after, dim=1)
    assert cosine.mean() > 0.99, f"mean feature cosine {cosine.mean():.4f}"
    assert score_before.shape == score_after.shape


@pytest.mark.cuda
def test_triton_kernel_matches_reference_on_cuda():
    from flashim import ops

    torch.manual_seed(5)
    for b, c, h, w in [(1, 32, 128, 128), (2, 64, 33, 29), (1, 128, 8, 8)]:
        x = torch.randn(b, c, h, w, device="cuda")
        offset = (torch.rand(b, 18, h, w, device="cuda") - 0.5) * 6.0
        scale = torch.tensor(0.05, device="cuda")
        reference = _deform_patches_int8_reference(x, offset, 3, 1, scale)
        dispatched = ops.deform_patches_int8(x.contiguous(), offset.contiguous(), 3, 1, scale)
        diff = (reference.values.int() - dispatched.values.int()).abs()
        assert diff.max() <= 1  # a single rounding-boundary quantum, same class as the int8-fast GEMM kernels
        assert (diff > 0).float().mean() < 1e-4
