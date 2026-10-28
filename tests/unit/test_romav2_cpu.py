"""FP8 presets must run on platforms without MXFP8 hardware (emulation kernels + references)."""

import pytest

from ..conftest import ASSETS

pytestmark = [pytest.mark.weights, pytest.mark.slow]


def test_fp8_emulation_on_cpu_tracks_bf16(fp32_highest):
    import flashim
    from flashim.ops import call_counts, reset_call_counts

    pair = (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")
    reset_call_counts()
    fp8 = flashim.load("romav2", "fp8", device="cpu", overrides={"arch.setting": "turbo"})
    assert set(fp8.quant_report.kernel_counts()) == {"reference"}
    bf16 = flashim.load("romav2", "bf16", device="cpu", overrides={"arch.setting": "turbo"})
    a, b = fp8.match_dense(*pair), bf16.match_dense(*pair)
    counts = call_counts()
    assert counts["depthwise5x5_relu_mxfp8"] == {"reference": 18}
    confident = b.overlap_AB[0, ..., 0] > 0.5
    err_px = ((a.warp_AB - b.warp_AB)[0][confident].norm(dim=-1) * 320 / 2).median()
    assert err_px < 1.0, f"median warp disagreement {err_px:.3f}px"
