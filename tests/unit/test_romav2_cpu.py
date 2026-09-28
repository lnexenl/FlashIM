"""FP8 presets must run on platforms without MXFP8 hardware (emulation kernels + references)."""

import pytest
import torch

from ..conftest import ASSETS

pytestmark = [pytest.mark.weights, pytest.mark.slow]


def test_fp8_emulation_on_cpu_tracks_bf16(fp32_highest):
    from flashim.models.romav2.build import load_romav2
    from flashim.ops import call_counts, reset_call_counts

    pair = (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")
    reset_call_counts()
    fp8 = load_romav2("fp8", setting="turbo", device="cpu")
    assert set(fp8.quant_report.kernel_counts()) == {"reference"}
    bf16 = load_romav2("bf16", setting="turbo", device="cpu")
    with torch.inference_mode():
        a = fp8.match(*pair)
        b = bf16.match(*pair)
    counts = call_counts()
    assert counts["depthwise5x5_relu_mxfp8"] == {"reference": 18}
    confident = b.overlap_AB[0, ..., 0] > 0.5
    err_px = ((a.warp_AB - b.warp_AB)[0][confident].norm(dim=-1) * 320 / 2).median()
    assert err_px < 1.0, f"median warp disagreement {err_px:.3f}px"
