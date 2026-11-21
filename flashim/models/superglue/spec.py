"""SuperGlue's :class:`~flashim.build.ModelSpec` (variants ``outdoor`` / ``indoor``; presets ``fp32`` / ``bf16``) and
the ``superpoint-superglue`` pipeline. See :mod:`flashim.models.superglue.quant` for measured INT8 / FP8 evidence."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from torch import nn

from flashim.build import PRECISIONS, BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.superglue import quant, weights
from flashim.models.superglue.config import SuperGlueConfig
from flashim.models.superglue.model import MultiHeadedAttention, SuperGlue


def _validate_variant(variant: str | None) -> str:
    v = variant or weights.DEFAULT_VARIANT
    if v not in weights.VARIANTS:
        raise ValueError(f"Unknown SuperGlue variant {v!r}; choose from {weights.VARIANTS}")
    return v


def _load(config: RunConfig) -> SuperGlue:
    variant = _validate_variant(config.variant)
    arch = dataclasses.replace(SuperGlueConfig(), **config.arch)
    return SuperGlue.from_pretrained(arch, variant=variant, checkpoint=config.checkpoint, device=config.device)


def _validate(config: RunConfig) -> None:
    _validate_variant(config.variant)


def _fold_batchnorm(model: SuperGlue, ctx: BuildContext) -> list[str]:
    return model.fold_batchnorm()


def _set_precision(model: SuperGlue, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


def _set_attention(model: nn.Module, impl: str) -> None:
    for module in model.modules():
        if isinstance(module, MultiHeadedAttention):
            module.sdpa_impl = impl


def _apply_inference(model: SuperGlue, inference: Mapping[str, Any]) -> None:
    if "sinkhorn_iterations" in inference:
        iters = int(inference["sinkhorn_iterations"])
        if iters < 0:
            raise ValueError(f"sinkhorn_iterations must be non-negative, got {iters}")
        model.sinkhorn_iterations = iters
    if "match_threshold" in inference:
        model.match_threshold = float(inference["match_threshold"])
    elif "threshold" in inference:
        model.match_threshold = float(inference["threshold"])


def _regions(model: SuperGlue) -> list[str]:
    return ["kenc", *(f"gnn.layers.{i}" for i in range(len(model.gnn.layers))), "final_proj"]


SPEC = ModelSpec(
    name="superglue",
    role="matcher",
    description="SuperGlue sparse matcher: SparseFeatures x SparseFeatures -> matches (variant outdoor | indoor)",
    load=_load,
    presets={
        "fp32": {
            "optimize": {"passes": ["fold_batchnorm"]},
            "inference": {"sinkhorn_iterations": 100, "match_threshold": 0.2},
        },
        "bf16": {"extends": "fp32", "precision": {"compute_dtype": "bf16"}},
    },
    default_preset="bf16",
    protected=quant.PROTECTED,
    passes={"fold_batchnorm": Pass(_fold_batchnorm, stage="pre_quant", description="exact linear+BN1d fold")},
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
    apply_inference=_apply_inference,
    validate=_validate,
)


# -- SuperPoint + SuperGlue pipeline -----------------------------------------------------------------------


def _compose_pipeline(config: RunConfig, build) -> nn.Module:
    from flashim.models.lightglue.pipeline import SparsePipeline

    matcher_cfg = config.components["matcher"]
    if config.variant is not None:
        matcher_cfg = matcher_cfg.replace(variant=config.variant)
    return SparsePipeline(build(config.components["extractor"]), build(matcher_cfg))


def _validate_pipeline(config: RunConfig) -> None:
    if config.variant is not None:
        _validate_variant(config.variant)


def _apply_pipeline_inference(pipeline: nn.Module, inference: Mapping[str, Any]) -> None:
    _apply_inference(pipeline.matcher, inference)


SUPERPOINT_SUPERGLUE = ModelSpec(
    name="superpoint-superglue",
    role="pipeline",
    description="SuperPoint (nms=4, threshold=0.005) + SuperGlue (Magic Leap weights; variant outdoor | indoor)",
    load=None,
    presets={
        preset: {
            "components": {
                "extractor": {
                    "model": "superpoint",
                    "preset": preset,
                    "arch": {"nms_radius": 4, "detection_threshold": 0.005, "remove_borders": 4},
                },
                "matcher": {"model": "superglue", "preset": preset, "variant": weights.DEFAULT_VARIANT},
            },
            "inference": {"max_matches": 1024, "threshold": 0.2},
        }
        for preset in PRECISIONS
    },
    default_preset="bf16",
    compose=_compose_pipeline,
    apply_inference=_apply_pipeline_inference,
    validate=_validate_pipeline,
)
