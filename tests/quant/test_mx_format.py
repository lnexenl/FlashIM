import pytest
import torch
from torch import nn

from flashim.ops.mx import _dequantize_swizzled
from flashim.quant.calibration import CalibrationTable, Calibrator, import_romav2_calibration
from flashim.quant.formats import mx
from flashim.quant.tensor import QuantizedWeight, quantize_static

from ..conftest import ROMAV2_ROOT


def test_exponent_rules():
    amax = torch.tensor([0.0, 448.0, 449.0, 1.0, 1e-30])
    e = mx.scale_exponent_from_amax(amax)
    assert e.dtype == torch.int8
    assert e.tolist() == [-127, 0, 1, -8, -108]  # ceil(log2(amax / 448)); only exact 0 -> -127


@pytest.mark.parametrize("channels", [32, 1000, 1024])
def test_static_quantize_roundtrip_and_saturation(channels):
    x = torch.randn(7, 5, channels) * 10
    exponent = mx.calibrate_block32_exponent(x)
    q = quantize_static(x, exponent)
    assert q.values.shape == (7, 5, mx.padded_channels(channels))
    back = q.dequantize()
    assert back.shape == x.shape
    # E4M3: half-ulp relative error 2^-4 for normals, absolute 2^-10 * scale for subnormals.
    scale = torch.exp2(exponent.float()).repeat_interleave(32)[:channels]
    assert ((back - x).abs() <= x.abs() * 2**-4 + scale * 2**-10 + 1e-12).all()
    # Values beyond the calibrated range saturate instead of overflowing.
    big = quantize_static(x * 1000, exponent).dequantize()
    assert torch.isfinite(big).all()
    assert (big.abs() <= 448 * torch.exp2(exponent.float()).repeat_interleave(32)[:channels] + 1e-3).all()


def test_weight_quantization_padding():
    w = torch.randn(100, 70)
    qw = QuantizedWeight.from_float(w)
    assert qw.values.shape == (128, 96) and qw.scale_exponent.shape == (128, 3)
    scale = torch.exp2(qw.scale_exponent.float()).repeat_interleave(32, dim=1)[:100, :70]
    assert ((qw.dequantize() - w).abs() <= w.abs() * 2**-4 + scale * 2**-10 + 1e-12).all()


@pytest.mark.parametrize("rows,groups", [(1, 1), (300, 32), (128, 4), (5000, 24)])
def test_swizzle_is_invertible(rows, groups):
    exponent = torch.randint(-20, 10, (rows, groups), dtype=torch.int8)
    values = torch.randn(rows, groups * 32).clamp(-448, 448).to(torch.float8_e4m3fn)
    dense = _dequantize_swizzled(values, mx.swizzle_scales(exponent))
    expected = values.float() * torch.exp2(exponent.float()).repeat_interleave(32, dim=1)
    assert torch.equal(dense, expected)


def test_calibrator_and_table_roundtrip(tmp_path):
    model = nn.Sequential(nn.Linear(40, 64), nn.GELU(), nn.Linear(64, 8))
    with Calibrator(model, ["0", "2"]) as calibrator:
        for _ in range(3):
            model(torch.randn(10, 40) * 5)
    table = calibrator.table({"note": "test"})
    assert table["0"].input_exponent.shape == (2,)
    assert table["2"].input_exponent.shape == (2,)
    path = tmp_path / "cal.safetensors"
    table.save(path)
    loaded = CalibrationTable.load(path)
    assert loaded.metadata["note"] == "test"
    for name in ("0", "2"):
        assert torch.equal(loaded[name].input_exponent, table[name].input_exponent)


ROMAV2_CAL = ROMAV2_ROOT / "data/quantization/romav2_full_fp8_mega1500.pt"


@pytest.mark.skipif(not ROMAV2_CAL.exists(), reason="RoMaV2 calibration file not present")
def test_import_romav2_calibration_is_exact(tmp_path):
    payload = torch.load(ROMAV2_CAL, weights_only=True)
    table = import_romav2_calibration(ROMAV2_CAL)
    table.save(tmp_path / "c.safetensors")
    loaded = CalibrationTable.load(tmp_path / "c.safetensors")
    assert set(loaded.linear) == set(payload["linear"])
    for name, value in payload["linear"].items():
        assert torch.equal(loaded[name].input_exponent, value["input_scale_exponent"].to(torch.int8))
        assert torch.equal(loaded[name].output_exponent, value["output_scale_exponent"].to(torch.int8))
