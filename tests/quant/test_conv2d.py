"""Static INT8 / MXFP8 convolutions (im2col + GEMM) against ``F.conv2d``."""

import pytest
import torch
from torch import nn

from flashim import ops
from flashim.quant.calibration import Calibrator, Int8Calibrator
from flashim.quant.methods.conv2d import StaticInt8Conv2d, StaticMXFP8Conv2d, im2col_bytes

pytestmark = pytest.mark.cuda

CASES = [  # in, out, kernel, stride, padding, dilation
    (64, 64, 3, 1, 1, 1),
    (48, 96, 3, 1, 1, 1),  # channels not a multiple of 32
    (32, 128, 7, 2, 3, 1),
    (128, 128, 1, 1, 0, 1),
    (64, 64, 3, 2, 1, 1),
    (32, 32, 3, 1, 2, 2),
]


def test_im2col_matches_unfold():
    x = torch.randint(-127, 127, (2, 13, 17, 5), dtype=torch.int8, device="cuda")
    cols, (ho, wo) = im2col_bytes(x, 3, 2, 1, 1)
    ref = torch.nn.functional.unfold(x.permute(0, 3, 1, 2).float(), 3, padding=1, stride=2)  # (c, ky, kx) order
    ref = ref.reshape(2, 5, 9, -1).permute(0, 3, 2, 1).reshape(2 * ho * wo, -1)  # -> (ky, kx, c)
    assert torch.equal(cols.float(), ref)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("fmt", ["int8", "fp8"])
def test_conv_matches_float(case, fmt):
    cin, cout, k, s, p, d = case
    torch.manual_seed(0)
    conv = nn.Conv2d(cin, cout, k, s, p, d).cuda().eval()
    x = torch.randn(2, cin, 40, 56, device="cuda")
    observer = Int8Calibrator(nn.Sequential(conv), ["0"]) if fmt == "int8" else Calibrator(nn.Sequential(conv), ["0"])
    net = nn.Sequential(conv)
    with torch.inference_mode(), observer:
        ref = net(x)
    table = observer.table()
    packed = (
        StaticInt8Conv2d.from_conv(conv, table["0"]) if fmt == "int8"
        else StaticMXFP8Conv2d.from_conv(conv, table["0"].input_exponent)
    )
    with torch.inference_mode():
        out = packed(x.bfloat16()).float()
    assert out.shape == ref.shape
    rel = (out - ref).norm() / ref.norm()
    assert rel < (0.02 if fmt == "int8" else 0.06), float(rel)


@pytest.mark.parametrize("cin,cout,k", [(256, 256, 3), (224, 224, 3), (512, 512, 3), (128, 96, 5)])  # 72 / 63 / 144 / 100 dots
@pytest.mark.parametrize("relu,residual", [(False, False), (True, True)])
def test_mxfp8_conv_fused_matches_im2col_above_the_ceiling(cin, cout, k, relu, residual):
    """Layers above 36 dot calls: the fused kernel (where a platform uses it for them, e.g. sm_89) against im2col + GEMM.
    Same GEMM bits; the fused epilogue adds bias / residual before its single BF16 rounding, the unfused one after."""
    torch.manual_seed(1)
    conv = nn.Conv2d(cin, cout, k, 1, k // 2).cuda().eval()
    x = torch.randn(1, cin, 44, 60, device="cuda")
    net = nn.Sequential(conv)
    observer = Calibrator(net, ["0"])
    with torch.inference_mode(), observer:
        net(x)
    packed = StaticMXFP8Conv2d.from_conv(conv, observer.table()["0"].input_exponent)
    skip = torch.randn(1, cout, 44, 60, device="cuda", dtype=torch.bfloat16) if residual else None
    outputs = {}
    with torch.inference_mode():
        for use_fused in (False, True):
            packed.use_fused = use_fused
            outputs[use_fused] = packed(x.bfloat16(), relu=relu, residual=skip).float()
    # two (three with a residual) BF16 roundings against one: a few ulp of the largest term
    scale = outputs[False].abs().max()
    assert (outputs[True] - outputs[False]).abs().max() <= scale * 2**-6
    assert ((outputs[True] - outputs[False]).norm() / outputs[False].norm()) < 5e-3


@pytest.mark.parametrize("cin,mid,cout,k", [(64, 64, 64, 3), (32, 48, 96, 3), (128, 256, 256, 1), (256, 256, 128, 3)])
@pytest.mark.parametrize("relu,residual", [(True, False), (False, False), (True, True)])
def test_mxfp8_conv_chain_bit_identical(cin, mid, cout, k, relu, residual):
    """A conv that writes the next conv's quantized input from its epilogue (``quantize_for``) gives the bytes of the
    separate quantize pass, so the pair's output does not change."""
    torch.manual_seed(2)
    first, second = nn.Conv2d(cin, mid, k, 1, k // 2).cuda().eval(), nn.Conv2d(mid, cout, 3, 1, 1).cuda().eval()
    net = nn.Sequential(first, nn.ReLU() if relu else nn.Identity(), second)
    x = torch.randn(2, cin, 37, 52, device="cuda")
    observer = Calibrator(net, ["0", "2"])
    with torch.inference_mode(), observer:
        net(x)
    table = observer.table()
    a = StaticMXFP8Conv2d.from_conv(first, table["0"].input_exponent)
    b = StaticMXFP8Conv2d.from_conv(second, table["2"].input_exponent)
    if not a.feeds(b):
        pytest.skip("conv chaining is not enabled on this platform")
    skip = torch.randn(2, mid, 37, 52, device="cuda", dtype=torch.bfloat16) if residual else None
    with torch.inference_mode():
        chained = a(x.bfloat16(), relu=relu, residual=skip, quantize_for=b)
        plain = a(x.bfloat16(), relu=relu, residual=skip)
        quantized = ops.quantize_mxfp8(plain.permute(0, 2, 3, 1), b.channel_exponent)
        assert torch.equal(chained.values.view(torch.uint8), quantized.values.view(torch.uint8))
        assert torch.equal(b(chained), b(plain))


@pytest.mark.parametrize("cin,mid,cout,k", [(64, 64, 64, 3), (20, 196, 128, 3), (128, 256, 256, 1)])
@pytest.mark.parametrize("relu,residual", [(True, False), (False, False), (True, True)])
def test_int8_conv_chain_bit_identical(cin, mid, cout, k, relu, residual):
    """INT8 counterpart of ``test_mxfp8_conv_chain_bit_identical``."""
    torch.manual_seed(3)
    first, second = nn.Conv2d(cin, mid, k, 1, k // 2).cuda().eval(), nn.Conv2d(mid, cout, 3, 1, 1).cuda().eval()
    net = nn.Sequential(first, nn.ReLU() if relu else nn.Identity(), second)
    x = torch.randn(2, cin, 37, 52, device="cuda")
    observer = Int8Calibrator(net, ["0", "2"])
    with torch.inference_mode(), observer:
        net(x)
    table = observer.table()
    a, b = StaticInt8Conv2d.from_conv(first, table["0"]), StaticInt8Conv2d.from_conv(second, table["2"])
    if not a.feeds(b):
        pytest.skip("conv chaining is not enabled on this platform")
    skip = torch.randn(2, mid, 37, 52, device="cuda", dtype=torch.bfloat16) if residual else None
    with torch.inference_mode():
        chained = a(x.bfloat16(), relu=relu, residual=skip, quantize_for=b)
        plain = a(x.bfloat16(), relu=relu, residual=skip)
        assert torch.equal(chained.values, ops.quantize_int8(plain.permute(0, 2, 3, 1), b.input_scale).values)
        assert torch.equal(b(chained), b(plain))
