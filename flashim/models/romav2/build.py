"""Build a ready-to-run RoMaV2 from a preset, a setting and an engine."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from flashim.engines.compile import compile_regions
from flashim.models.romav2.config import RoMaV2Config
from flashim.models.romav2.model import RoMaV2
from flashim.models.romav2.quant import DEFAULT_CALIBRATION, PRESETS, PROTECTED
from flashim.platforms import Feature, platform_for
from flashim.quant.apply import quantize_model
from flashim.quant.calibration import CalibrationTable
from flashim.quant.passes.refiner_block import fold_refiner_blocks

logger = logging.getLogger(__name__)

COMPILE_REGIONS = ("f", "matcher.mv_vit")


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
) -> RoMaV2:
    """Load RoMaV2 weights, apply a quantization preset and an execution engine.

    ``engine``: ``"eager"`` or ``"compile"`` (``torch.compile`` of the DINOv3 and
    matcher-ViT regions). On platforms without native MXFP8 GEMMs, FP8 presets run
    with emulation kernels (``strict=True`` raises instead).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if engine not in ("eager", "compile"):
        raise ValueError(f"Unknown engine {engine!r}")
    cfg = config or RoMaV2Config(setting=setting)
    model = RoMaV2.from_pretrained(cfg, device=device)
    platform = platform_for(model.device)
    spec = PRESETS[preset]
    model.quant_report = None
    model.preset = preset
    if spec.config is not None:
        if not platform.supports(Feature.MXFP8_SCALED_MM):
            message = f"Preset {preset!r}: no native MXFP8 GEMM on {platform.describe()}; using emulation"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
        table = calibration
        if not isinstance(table, CalibrationTable):
            table = CalibrationTable.load(table or DEFAULT_CALIBRATION)
        if spec.fold_refiners:
            fold_refiner_blocks(model.refiners, channels_last=True)
        model.quant_report = quantize_model(model, spec.config, table, protected=PROTECTED, platform=platform)
        model.calibration_metadata = dict(table.metadata)
    if engine == "compile":
        compile_regions(model, COMPILE_REGIONS, mode=compile_mode)
    model.engine = engine
    return model
