"""Build a ready-to-run RDD from a preset and an engine."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from flashim.engines.compile import compile_regions
from flashim.layers import MSDeformAttn
from flashim.models.rdd.config import RDDConfig
from flashim.models.rdd.model import RDD
from flashim.models.rdd.quant import PRESETS, PROTECTED
from flashim.platforms import Feature, platform_for
from flashim.quant.apply import quantize_model
from flashim.quant.calibration import CalibrationTable

logger = logging.getLogger(__name__)

ENGINES = ("eager", "compile")
RECOMPILE_LIMIT = 64
# Inductor fuses the norms/activations/permutes around the (custom-op) GEMMs and the deformable kernel.
COMPILE_REGIONS = ("backbone", *(f"input_proj.{i}" for i in range(5)), "detector", "transformer")


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
    """Load RDD weights, apply a precision/quantization preset and an execution engine.

    ``engine="compile"`` compiles the backbone, input projections, detector and transformer once per input
    resolution (``compile_mode=None`` compiles ~5x faster than max-autotune for ~3% more runtime).
    On platforms without native MXFP8 GEMMs FP8 presets run with emulation kernels (``strict=True`` raises
    instead). ``detector_dtype`` overrides the preset's keypoint-detector precision (its score head stays FP32).
    ``checkpoint``: path to ``rdd.pth`` (see :mod:`flashim.models.rdd.weights` for the search order).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}; choose from {ENGINES}")
    spec = PRESETS[preset]
    model = RDD.from_pretrained(config, checkpoint=checkpoint, device=device)
    model.preset = preset
    if spec.compute_dtype != torch.float32:
        model.set_compute_dtype(spec.compute_dtype)
    if isinstance(detector_dtype, str):
        detector_dtype = getattr(torch, detector_dtype)
    detector_dtype = detector_dtype or spec.detector_dtype
    if detector_dtype != torch.float32:
        model.detector.set_dtype(detector_dtype)
    if spec.config is not None:
        platform = platform_for(model.device)
        if not platform.supports(Feature.MXFP8_SCALED_MM):
            message = f"Preset {preset!r}: no native MXFP8 GEMM on {platform.describe()}; using emulation"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
        table = calibration
        if not isinstance(table, CalibrationTable):
            table = CalibrationTable.load(table or spec.calibration)
        model.quant_report = quantize_model(model, spec.config, table, protected=PROTECTED)
        model.calibration_metadata = dict(table.metadata)
    fused = [m.fuse_query_projections() for m in model.modules() if isinstance(m, MSDeformAttn)]
    model.fused_query_projections = sum(fused)
    if engine == "compile":
        # Regions compile once per input resolution (static shapes); a benchmark with many aspect ratios
        # (MegaDepth-1500 at 1600px: 24) would otherwise exceed Dynamo's default limit of 8 and silently
        # run the remaining resolutions eagerly.
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, RECOMPILE_LIMIT)
        compile_regions(model, COMPILE_REGIONS, mode=compile_mode)
    model.engine = engine
    return model
