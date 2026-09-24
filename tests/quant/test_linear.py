import pytest
import torch
from torch import nn

from flashim.platforms import Platform, platform_for
from flashim.quant.formats import mx
from flashim.quant.methods.linear import StaticMXFP8Linear


def _layer(out_f, in_f, platform, device):
    torch.manual_seed(0)
    linear = nn.Linear(in_f, out_f).to(device)
    calib = torch.randn(64, in_f, device=device) * 3
    return linear, StaticMXFP8Linear.from_linear(linear, mx.calibrate_block32_exponent(calib), platform)


def test_emulation_on_cpu_close_to_dense():
    linear, packed = _layer(96, 160, Platform("cpu"), "cpu")
    assert packed.kernel.name == "emulation"
    x = torch.randn(4, 9, 160) * 3
    out = packed(x)
    assert out.shape == (4, 9, 96) and out.dtype == x.dtype
    ref = linear(x)
    assert ((out - ref).norm() / ref.norm()) < 0.05


@pytest.mark.cuda
@pytest.mark.sm120
def test_kernel_choice_and_agreement():
    platform = platform_for("cuda")
    lin, native = _layer(768, 1024, platform, "cuda")
    _, triton = _layer(96, 160, platform, "cuda")
    assert native.kernel.name == "cublas_mxfp8"
    assert triton.kernel.name == "triton_fp8"
    emulated = StaticMXFP8Linear.from_linear(lin, native.input_exponent, Platform("cpu")).cuda()
    x = torch.randn(300, 1024, device="cuda", dtype=torch.bfloat16)
    a, b = native(x), emulated(x)
    assert ((a.float() - b.float()).norm() / b.float().norm()) < 1e-3


@pytest.mark.cuda
@pytest.mark.sm120
def test_prequantized_requires_same_scale():
    from flashim.quant.tensor import quantize_static

    _, native = _layer(768, 1024, platform_for("cuda"), "cuda")
    x = torch.randn(10, 1024, device="cuda")
    out = native.forward_prequantized(quantize_static(x, native.input_exponent))
    assert out.dtype == torch.bfloat16 and out.shape == (10, 768)
    with pytest.raises(ValueError):
        native.forward_prequantized(quantize_static(x, native.input_exponent + 1))
