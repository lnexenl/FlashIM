from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from flashim import ops
from flashim.paths import (  # noqa: F401  (re-exported to the tests)
    ASSETS,
    MEGADEPTH_IMAGES,
    ORACLES,
    ROMAV2_CALIBRATION,
)

# the RoMaV2 reference package (vendored) shadows any installed copy
sys.path.insert(0, str(ORACLES / "romav2"))

CHECKPOINT = Path(torch.hub.get_dir()) / "checkpoints" / "romav2.0.1.pt"

HAS_CUDA = torch.cuda.is_available()
CAPABILITY = torch.cuda.get_device_capability() if HAS_CUDA else (0, 0)


def pytest_collection_modifyitems(config, items):
    skips = {
        "cuda": (not HAS_CUDA, "needs CUDA"),
        "sm120": (CAPABILITY < (10, 0), "needs native MXFP8 scaled_mm (sm_100+)"),
        "weights": (not CHECKPOINT.exists(), f"needs {CHECKPOINT}"),
        "oracle": (importlib.util.find_spec("romav2") is None, "needs the romav2 reference package"),
    }
    for item in items:
        for marker, (skip, reason) in skips.items():
            if skip and marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _reset_op_overrides():
    yield
    ops.configure(None)
    ops.set_strict(False)


_BACKEND_FLAGS = [
    (torch.backends.cuda.flash_sdp_enabled, torch.backends.cuda.enable_flash_sdp),
    (torch.backends.cuda.mem_efficient_sdp_enabled, torch.backends.cuda.enable_mem_efficient_sdp),
    (torch.backends.cuda.math_sdp_enabled, torch.backends.cuda.enable_math_sdp),
    (torch.backends.cuda.cudnn_sdp_enabled, torch.backends.cuda.enable_cudnn_sdp),
    (lambda: torch.backends.cudnn.deterministic, lambda v: setattr(torch.backends.cudnn, "deterministic", v)),
    (lambda: torch.backends.cudnn.benchmark, lambda v: setattr(torch.backends.cudnn, "benchmark", v)),
]


@pytest.fixture(autouse=True)
def _restore_torch_backends():
    """The vendored LightGlue oracles set ``cudnn.deterministic`` on import and call ``enable_flash_sdp(False)``
    in ``Attention.__init__``: without this, every later test runs SDPA without flash attention (measured: it flips
    ``test_reduced_precision_tracks_fp32[fp8]`` from 553/617 to 559/617 shared matches)."""
    previous = [get() for get, _ in _BACKEND_FLAGS]
    yield
    for (_, set_), value in zip(_BACKEND_FLAGS, previous):
        set_(value)


@pytest.fixture
def fp32_highest():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)
