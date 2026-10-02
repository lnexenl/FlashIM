"""Static INT8 / MXFP8 convolutions (im2col + GEMM) against ``F.conv2d``."""

import pytest
import torch
from torch import nn

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
