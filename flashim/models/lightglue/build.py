"""Build a ready-to-run LightGlue (and the RDD + LightGlue pipeline) from a preset and an engine."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import torch

from flashim.engines.compile import compile_regions
from flashim.models.lightglue import weights
from flashim.models.lightglue.config import ALIKED_CONFIG, LightGlueConfig
from flashim.models.lightglue.model import CrossBlock, LightGlue, SelfBlock
from flashim.models.lightglue.pipeline import SparsePipeline
from flashim.models.lightglue.quant import PRESETS, PROTECTED
from flashim.platforms import Feature, platform_for
from flashim.quant.apply import quantize_model
from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable

logger = logging.getLogger(__name__)

ENGINES = ("eager", "compile")
RECOMPILE_LIMIT = 64


def load_lightglue(
    preset: str = "fp8",
    *,
    variant: str = "rdd",
    engine: str = "eager",
    device: torch.device | str | None = None,
    checkpoint: str | Path | None = None,
    calibration: str | Path | CalibrationTable | Int8CalibrationTable | None = None,
    strict: bool = False,
    config: LightGlueConfig | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    attention: str = "sdpa",
    depth_confidence: float | None = None,
) -> LightGlue:
    """Load LightGlue weights, apply a precision/quantization preset and an execution engine.

    ``variant``: ``"rdd"`` (default, ``rdd_lg.pth``, dim 256 throughout) or ``"aliked"`` (cvg/LightGlue's
    ``raco_aliked_lightglue.pth``, ALIKED's 128-d descriptors projected to a 256-wide transformer — see
    :data:`flashim.models.lightglue.config.ALIKED_CONFIG`). ``engine="compile"`` compiles the positional
    encoding, every transformer layer and the assignment head once per keypoint count. ``attention``:
    ``"sdpa"`` (torch SDPA) or ``"sage"`` / ``"sage-fp8"`` (SageAttention INT8-QK attention; needs the
    ``sageattention`` package). On platforms without native MXFP8 GEMMs the FP8 preset runs with emulation
    kernels (``strict=True`` raises instead). ``depth_confidence`` in (0, 1) enables early stopping
    (LightGlue recommends 0.95): layers stop once that fraction of keypoints is confident, so with
    ``engine="compile"`` each executed depth reuses the per-layer graphs (one host sync per layer).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}; choose from {ENGINES}")
    if variant not in ("rdd", "aliked"):
        raise ValueError(f"Unknown variant {variant!r}; choose from ('rdd', 'aliked')")
    spec = PRESETS[preset]
    config = config or (ALIKED_CONFIG if variant == "aliked" else LightGlueConfig())
    if depth_confidence is not None:
        config = dataclasses.replace(config, depth_confidence=depth_confidence)
    if variant == "aliked":
        state = weights.load_aliked_state_dict(config.num_heads, config.num_layers, checkpoint)
        model = LightGlue.from_pretrained(config, state_dict=state, device=device)
    else:
        model = LightGlue.from_pretrained(config, checkpoint=checkpoint, device=device)
    model.preset = preset
    if spec.compute_dtype != torch.float32:
        model.set_compute_dtype(spec.compute_dtype)
    if spec.config is not None:
        platform = platform_for(model.device)
        if not spec.int8 and not platform.supports(Feature.MXFP8_SCALED_MM):
            message = f"Preset {preset!r}: no native MXFP8 GEMM on {platform.describe()}; using emulation"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
        table = calibration
        if not isinstance(table, (CalibrationTable, Int8CalibrationTable)):
            table = (Int8CalibrationTable if spec.int8 else CalibrationTable).load(table or spec.calibration)
        model.quant_report = quantize_model(model, spec.config, table, protected=PROTECTED, platform=platform)
        model.calibration_metadata = dict(table.metadata)
    if attention != "sdpa":
        for module in model.modules():
            if isinstance(module, (SelfBlock, CrossBlock)):
                module.sdpa_impl = attention
        model.attention = attention
    if engine == "compile":
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, RECOMPILE_LIMIT)
        regions = (
            "posenc",
            *(f"transformers.{i}" for i in range(len(model.transformers))),
            *(f"log_assignment.{i}" for i in range(len(model.log_assignment))),
        )
        compile_regions(model, regions, mode=compile_mode)
    model.engine = engine
    return model


def load_rdd_lightglue(
    preset: str = "fp8",
    *,
    rdd_preset: str = "fp8",
    engine: str = "eager",
    device: torch.device | str | None = None,
    rdd_kwargs: dict | None = None,
    **kwargs,
) -> SparsePipeline:
    """RDD keypoints/descriptors (``rdd_preset``) matched by LightGlue (``preset``); ``kwargs`` go to
    :func:`load_lightglue` and ``rdd_kwargs`` to :func:`flashim.models.rdd.build.load_rdd`."""
    from flashim.models.rdd.build import load_rdd

    extractor = load_rdd(rdd_preset, engine=engine, device=device, **(rdd_kwargs or {}))
    matcher = load_lightglue(preset, engine=engine, device=device, **kwargs)
    return SparsePipeline(extractor, matcher)


def load_aliked_lightglue(
    preset: str = "fp32",
    *,
    aliked_preset: str = "fp32",
    engine: str = "eager",
    device: torch.device | str | None = None,
    aliked_kwargs: dict | None = None,
    **kwargs,
) -> SparsePipeline:
    """ALIKED keypoints/descriptors (``aliked_preset``) matched by cvg/LightGlue's ALIKED-trained weights
    (``preset``); ``kwargs`` go to :func:`load_lightglue` (``variant="aliked"`` is set automatically) and
    ``aliked_kwargs`` to :func:`flashim.models.aliked.build.load_aliked`."""
    from flashim.models.aliked.build import load_aliked

    extractor = load_aliked(aliked_preset, engine=engine, device=device, **(aliked_kwargs or {}))
    matcher = load_lightglue(preset, variant="aliked", engine=engine, device=device, **kwargs)
    return SparsePipeline(extractor, matcher)


def load_raco_aliked_lightglue(
    preset: str = "fp32",
    *,
    extractor_preset: str = "fp32",
    engine: str = "eager",
    device: torch.device | str | None = None,
    extractor_kwargs: dict | None = None,
    **kwargs,
) -> SparsePipeline:
    """RaCo keypoints + ALIKED descriptors (``extractor_preset``) matched by cvg/LightGlue's ALIKED-trained
    weights (``preset``): the pipeline of `fabio-sim/LightGlue-ONNX`'s ``raco_aliked``. ``kwargs`` go to
    :func:`load_lightglue`, ``extractor_kwargs`` to :func:`flashim.models.raco.build.load_raco_aliked`."""
    from flashim.models.raco.build import load_raco_aliked

    extractor = load_raco_aliked(extractor_preset, engine=engine, device=device, **(extractor_kwargs or {}))
    matcher = load_lightglue(preset, variant="aliked", engine=engine, device=device, **kwargs)
    return SparsePipeline(extractor, matcher)
