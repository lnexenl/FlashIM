"""Build a ready-to-run ALIKED (``load_aliked``: the keyword API over :mod:`flashim.build`)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.build import build
from flashim.models.aliked.config import ALIKEDConfig
from flashim.models.aliked.model import ALIKED
from flashim.models.aliked.spec import DEFAULT_VARIANT, legacy_overrides
from flashim.quant.calibration import Int8CalibrationTable


def load_aliked(
    preset: str = "fp32",
    *,
    variant: str = DEFAULT_VARIANT,
    engine: str = "eager",
    device: torch.device | str | None = None,
    checkpoint: str | Path | None = None,
    config: ALIKEDConfig | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    channels_last: bool = False,
    deform_int8_calibration: str | Path | Int8CalibrationTable | None = None,
) -> ALIKED:
    """Load ALIKED weights and apply a precision preset and an execution engine (see ``spec.py``).

    ``variant``: one of ``aliked-t16`` / ``aliked-n16`` / ``aliked-n16rot`` / ``aliked-n32``; ``config`` overrides
    the variant's architecture. Every preset folds each block's eval BatchNorm into its convolution (exact; ~15-30%
    faster on an RTX 5070 Ti). ``channels_last`` (opt-in: measured slower here, ALIKED's channels are tiny and
    ``deform_conv2d`` returns NCHW) and ``deform_int8_calibration`` (INT8 deformable convs, a win at batch >= 4)
    add passes. ``engine="compile"`` compiles the encoder blocks and the descriptor head once per resolution.
    Equivalent to ``flashim.build("aliked", preset, config=...)``.
    """
    return build(
        "aliked",
        preset,
        config=legacy_overrides(
            variant=variant, engine=engine, device=device, checkpoint=checkpoint, config=config,
            compile_mode=compile_mode, channels_last=channels_last, deform_int8_calibration=deform_int8_calibration,
        ),
    )
