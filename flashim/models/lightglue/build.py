"""Build LightGlue and the sparse pipelines around it (the keyword API over :mod:`flashim.build`)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.build import build
from flashim.models.lightglue.config import LightGlueConfig
from flashim.models.lightglue.model import LightGlue
from flashim.models.lightglue.pipeline import SparsePipeline
from flashim.models.lightglue.spec import legacy_overrides
from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable


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

    ``variant``: ``"rdd"`` (default, ``rdd_lg.pth``, dim 256 throughout), ``"aliked"`` (cvg/LightGlue's
    ``aliked_lightglue.pth``, trained on ALIKED's keypoints) or ``"raco-aliked"`` (``raco_aliked_lightglue.pth``,
    trained on RaCo's keypoints), both projecting ALIKED's 128-d descriptors to a 256-wide transformer — see
    :data:`flashim.models.lightglue.config.ALIKED_CONFIG`. ``engine="compile"`` compiles the positional
    encoding, every transformer layer and the assignment head once per keypoint count. ``attention``:
    ``"sdpa"`` (torch SDPA) or ``"sage"`` / ``"sage-fp8"`` (SageAttention INT8-QK attention; needs the
    ``sageattention`` package). On a GPU without FP8 (below sm_89) the FP8 preset is replaced by ``int8``; on the CPU
    it runs with emulation kernels (``strict=True`` raises instead of either). ``depth_confidence`` in (0, 1) enables early stopping
    (LightGlue recommends 0.95): layers stop once that fraction of keypoints is confident, so with
    ``engine="compile"`` each executed depth reuses the per-layer graphs (one host sync per layer).
    Equivalent to ``flashim.build("lightglue", preset, config=...)``.
    """
    return build(
        "lightglue",
        preset,
        config=legacy_overrides(
            variant=variant, engine=engine, device=device, checkpoint=checkpoint, calibration=calibration,
            strict=strict, config=config, compile_mode=compile_mode, attention=attention,
            depth_confidence=depth_confidence,
        ),
    )


def _pipeline(name: str, preset: str, extractor: dict, engine: str, device, kwargs: dict, variant: str) -> SparsePipeline:
    matcher = {"model": "lightglue", "preset": preset, **legacy_overrides(variant=variant, engine=engine, **kwargs)}
    overlay: dict = {"engine": {"kind": engine}, "components": {"extractor": extractor, "matcher": matcher}}
    if device is not None:
        overlay["device"] = str(device)
    return build(name, preset, config=overlay)


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
    from flashim.models.rdd.spec import legacy_overrides as rdd_overrides

    extractor = {"model": "rdd", "preset": rdd_preset, **rdd_overrides(engine=engine, **(rdd_kwargs or {}))}
    return _pipeline("rdd-lightglue", preset, extractor, engine, device, kwargs, "rdd")


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
    from flashim.models.aliked.spec import legacy_overrides as aliked_overrides

    extractor = {"model": "aliked", "preset": aliked_preset, **aliked_overrides(engine=engine, **(aliked_kwargs or {}))}
    return _pipeline("aliked-lightglue", preset, extractor, engine, device, kwargs, "aliked")


def load_raco_aliked_lightglue(
    preset: str = "fp32",
    *,
    extractor_preset: str = "fp32",
    engine: str = "eager",
    device: torch.device | str | None = None,
    extractor_kwargs: dict | None = None,
    **kwargs,
) -> SparsePipeline:
    """RaCo keypoints + ALIKED descriptors (``extractor_preset``) matched by cvg/LightGlue's weights trained on
    that pairing (``raco_aliked_lightglue.pth``, ``preset``): the pipeline of `fabio-sim/LightGlue-ONNX`'s
    ``raco_aliked``. ``kwargs`` go to
    :func:`load_lightglue`, ``extractor_kwargs`` (``raco_checkpoint``, ``raco_config``, ``aliked_kwargs``,
    ``fused_ranker``) to :func:`flashim.models.raco.build.load_raco_aliked`."""
    from flashim.models.raco.build import raco_aliked_overlay

    overlay = raco_aliked_overlay(extractor_preset, engine=engine, **(extractor_kwargs or {}))
    extractor = {"model": "raco-aliked", "preset": extractor_preset, **overlay}
    return _pipeline("raco-aliked-lightglue", preset, extractor, engine, device, kwargs, "raco-aliked")
