"""Build a ready-to-run RoMaV2 (``load_romav2``: the keyword API over :mod:`flashim.build`)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.build import build
from flashim.models.romav2.config import RoMaV2Config
from flashim.models.romav2.model import RoMaV2
from flashim.models.romav2.spec import COMPILE_FULL_REGIONS, COMPILE_REGIONS, legacy_overrides
from flashim.quant.calibration import CalibrationTable

__all__ = ["COMPILE_FULL_REGIONS", "COMPILE_REGIONS", "load_romav2"]


def load_romav2(
    preset: str = "fp8",
    *,
    setting: str = "precise",
    engine: str = "eager",
    device: torch.device | str | None = None,
    calibration: str | Path | CalibrationTable | None = None,
    strict: bool = False,
    config: RoMaV2Config | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    fast_sampling: bool | None = None,
    attention: str = "sdpa",
) -> RoMaV2:
    """Load RoMaV2 weights, apply a quantization preset and an execution engine.

    ``engine``:
      - ``"eager"``;
      - ``"compile"``: ``torch.compile`` of the DINOv3 and matcher-ViT regions (RoMaV2's
        validated configuration);
      - ``"compile-full"``: additionally folds VGG BatchNorm and compiles VGG, the DPT
        head and the refiner conv stacks (~20% faster; changes rounding vs ``compile``).
    On platforms without native MXFP8 GEMMs, FP8 presets run with emulation kernels
    (``strict=True`` raises instead).

    ``attention``: ``"sdpa"`` (torch SDPA, default), ``"sage"`` or ``"sage-fp8"`` (SageAttention INT8-QK attention
    in every quantized transformer block; needs the ``sageattention`` package; not RoMaV2-exact).

    ``fast_sampling``: fused-KDE ``sample()`` (default on; ``FLASHIM_FAST_SAMPLING=0`` restores RoMaV2's).
    Equivalent to ``flashim.build("romav2", preset, config=...)``.
    """
    return build(
        "romav2",
        preset,
        config=legacy_overrides(
            setting=setting, engine=engine, device=device, calibration=calibration, strict=strict, config=config,
            compile_mode=compile_mode, fast_sampling=fast_sampling, attention=attention,
        ),
    )
