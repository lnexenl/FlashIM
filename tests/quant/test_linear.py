import pytest
import torch
from torch import nn

from flashim import ops
from flashim.quant.formats import mx
from flashim.quant.methods.linear import StaticMXFP8Linear


def _layer(out_f, in_f, device, impl=None):
    torch.manual_seed(0)
    linear = nn.Linear(in_f, out_f).to(device)
    calib = torch.randn(64, in_f, device=device) * 3
    return linear, StaticMXFP8Linear.from_linear(linear, mx.calibrate_block32_exponent(calib), impl=impl)


def test_emulation_on_cpu_close_to_dense():
    linear, packed = _layer(96, 160, "cpu")
    assert packed.kernel_name == "reference"
    x = torch.randn(4, 9, 160) * 3
    out = packed(x)
    assert out.shape == (4, 9, 96) and out.dtype == x.dtype
    ref = linear(x)
    assert ((out - ref).norm() / ref.norm()) < 0.05


@pytest.mark.cuda
@pytest.mark.sm120
def test_kernel_choice_and_agreement():
    lin, native = _layer(768, 1024, "cuda")
    _, small = _layer(96, 160, "cuda")
    assert native.kernel_name == "cublas"
    assert small.kernel_name == "triton"  # cuBLAS needs N, K % 128 == 0: accepts() rules it out at bind time
    emulated = StaticMXFP8Linear.from_linear(lin, native.input_exponent, impl="reference")
    x = torch.randn(300, 1024, device="cuda", dtype=torch.bfloat16)
    ops.reset_call_counts()
    a, b = native(x), emulated(x)
    assert ops.call_counts()["mxfp8_linear"] == {"cublas": 1, "reference": 1}
    assert ((a.float() - b.float()).norm() / b.float().norm()) < 1e-3


@pytest.mark.cuda
@pytest.mark.sm120
def test_moving_the_layer_rebinds_its_gemm():
    lin, layer = _layer(256, 256, "cpu")
    assert layer.kernel_name == "reference"
    layer = layer.cuda()
    assert layer.kernel_name == "cublas"
    x = torch.randn(40, 256, device="cuda")
    ref = lin.cuda()(x)
    assert ((layer(x).float() - ref).norm() / ref.norm()) < 0.05


@pytest.mark.cuda
@pytest.mark.sm120
def test_prequantized_requires_same_scale():
    from flashim.quant.tensor import quantize_static

    _, native = _layer(768, 1024, "cuda")
    x = torch.randn(10, 1024, device="cuda")
    out = native.forward_prequantized(quantize_static(x, native.input_exponent))
    assert out.dtype == torch.bfloat16 and out.shape == (10, 768)
    with pytest.raises(ValueError):
        native.forward_prequantized(quantize_static(x, native.input_exponent + 1))
