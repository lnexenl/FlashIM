"""Launch-config selection of the Triton FP8 GEMM (CPU only): the 132-SM Hopper path must not leak to other devices."""

import torch

from flashim.backends.triton import gemm


def test_swapped_config_only_for_large_measured_shapes():
    assert gemm._swapped_config(8192, 512, 512) == (128, 128, 64, 8, 3)
    assert gemm._swapped_config(131072, 1024, 1536) == (128, 128, 64, 8, 3)
    assert gemm._swapped_config(4096, 512, 512) is None  # few rows: the old kernel
    assert gemm._swapped_config(8192, 256, 512) is None  # short K
    assert gemm._swapped_config(8192, 512, 96) is None  # N not a multiple of the tile


def test_sm90_config_is_unchanged():
    assert gemm._gemm_config_sm90(8192, 512, 512) == (64, 128, 32, 4, 3)
    assert gemm._gemm_config_sm90(131072, 1024, 1024) == (64, 128, 32, 4, 3)


def test_other_devices_never_take_the_swapped_path(monkeypatch):
    device = torch.device("cuda", 0)
    for capability, sms, expected in [((9, 0), 78, False), ((9, 0), 114, False), ((9, 0), 132, True), ((8, 9), 132, False),
                                      ((12, 0), 132, False)]:
        monkeypatch.setattr(gemm, "platform_for", lambda d, c=capability: type("P", (), {"capability": c})())
        monkeypatch.setattr(gemm, "_probe_sm_count", lambda d, n=sms: n)
        gemm._SM_COUNT.clear()
        assert gemm._is_sm90_132(device) is expected, (capability, sms)
    gemm._SM_COUNT.clear()
