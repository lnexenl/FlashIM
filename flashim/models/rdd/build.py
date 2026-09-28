"""Build a ready-to-run RDD (``load_rdd``: the keyword API over :mod:`flashim.build`)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.build import build
from flashim.models.rdd.config import RDDConfig
from flashim.models.rdd.model import RDD
from flashim.models.rdd.spec import COMPILE_REGIONS, legacy_overrides
from flashim.quant.calibration import CalibrationTable

__all__ = ["COMPILE_REGIONS", "load_rdd"]


def load_rdd(
    preset: str = "fp8",
    *,
    engine: str = "eager",
    device: torch.device | str | None = None,
    checkpoint: str | Path | None = None,
    calibration: str | Path | CalibrationTable | None = None,
    strict: bool = False,
    config: RDDConfig | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    detector_dtype: torch.dtype | str | None = None,
) -> RDD:
    """Load RDD weights, apply a precision/quantization preset and an execution engine (see ``quant.py``).

    ``engine="compile"`` compiles the backbone, input projections, detector and transformer once per input
    resolution (``compile_mode=None`` compiles ~5x faster than max-autotune for ~3% more runtime). On platforms
    without native MXFP8 GEMMs FP8 presets run with emulation kernels (``strict=True`` raises instead).
    ``detector_dtype`` overrides the keypoint-detector precision (its score head stays FP32). ``checkpoint``: path
    to ``rdd.pth`` (see :mod:`flashim.models.rdd.weights` for the search order). Equivalent to
    ``flashim.build("rdd", preset, config=...)``.
    """
    return build(
        "rdd",
        preset,
        config=legacy_overrides(
            engine=engine, device=device, checkpoint=checkpoint, calibration=calibration, strict=strict,
            config=config, compile_mode=compile_mode, detector_dtype=detector_dtype,
        ),
    )
