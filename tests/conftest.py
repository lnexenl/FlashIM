from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from flashim import ops
from flashim.paths import (  # noqa: F401  (re-exported to the tests)
    ASSETS,
    MEGADEPTH_IMAGES,
    ROMAV2_CALIBRATION,
)

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


@pytest.fixture
def fp32_highest():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)
