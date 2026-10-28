"""LightGlue's spec and the sparse pipelines built around it (RDD / ALIKED / RaCo + ALIKED / ... -> LightGlue).

A pipeline's preset is every component's: each builds the nearest preset it has (``aliked-lightglue`` ``int8``: ALIKED
``bf16``, LightGlue ``int8``; ``model.build_report.fallbacks`` lists them); ``components.extractor.preset`` /
``components.matcher.preset`` set one component's."""

from __future__ import annotations

import dataclasses

from torch import nn

from flashim.build import PRECISIONS, EngineSpec, ModelSpec, dtype
from flashim.config import RunConfig
from flashim.models.lightglue import quant, weights
from flashim.models.lightglue.config import (
    ALIKED_CONFIG,
    DISK_CONFIG,
    SUPERPOINT_CONFIG,
    XFEAT_CONFIG,
    LightGlueConfig,
)
from flashim.models.lightglue.model import CrossBlock, LightGlue, SelfBlock

VARIANTS = {
    "rdd": LightGlueConfig(),
    "aliked": ALIKED_CONFIG,
    "raco-aliked": ALIKED_CONFIG,
    "superpoint": SUPERPOINT_CONFIG,
    "disk": DISK_CONFIG,
    "xfeat": XFEAT_CONFIG,
}


def _load(config: RunConfig) -> LightGlue:
    variant = config.variant or "rdd"
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    arch = dataclasses.replace(VARIANTS[variant], **config.arch)
    if variant == "xfeat":
        from flashim.models.xfeat.weights import load_lighterglue_state_dict

        state = load_lighterglue_state_dict(arch.num_heads, arch.num_layers, config.checkpoint)
        return LightGlue.from_pretrained(arch, state_dict=state, device=config.device)
    if variant in weights.CVG_FILES:
        state = weights.load_cvg_state_dict(variant, arch.num_heads, arch.num_layers, config.checkpoint)
        return LightGlue.from_pretrained(arch, state_dict=state, device=config.device)
    return LightGlue.from_pretrained(arch, checkpoint=config.checkpoint, device=config.device)


def _set_precision(model: LightGlue, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


def _set_attention(model: nn.Module, impl: str) -> None:
    for module in model.modules():
        if isinstance(module, (SelfBlock, CrossBlock)):
            module.sdpa_impl = impl


def _regions(model: LightGlue) -> list[str]:
    return [
        "posenc",
        *(f"transformers.{i}" for i in range(len(model.transformers))),
        *(f"log_assignment.{i}" for i in range(len(model.log_assignment))),
    ]


_QUANTIZED = {name: p for name, p in quant.PRESETS.items() if p.config is not None}

SPEC = ModelSpec(
    name="lightglue",
    role="matcher",
    description="LightGlue sparse matcher: SparseFeatures x SparseFeatures -> matches (variant rdd | aliked | raco-aliked | superpoint | disk | xfeat)",
    load=_load,
    presets={
        "fp32": {},
        "bf16": {"precision": {"compute_dtype": "bf16"}},
        **{name: {"extends": "bf16", "precision": {"quant": name}} for name in _QUANTIZED},
    },
    default_preset="fp8",
    quant_configs={name: p.config for name, p in _QUANTIZED.items()},
    calibrations={
        **{name: p.calibration for name, p in _QUANTIZED.items()},
        # a variant's own table where one ships next to the default (RDD's) one
        **{f"{variant}/{name}": table for variant in VARIANTS for name in _QUANTIZED
           if (table := quant.CALIBRATION_DIR / f"lightglue_{variant}_{name}_mega.safetensors").exists()},
    },
    protected=quant.PROTECTED,
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
)


# -- sparse pipelines ---------------------------------------------------------------------------------------


def _compose(config: RunConfig, build) -> nn.Module:
    from flashim.models.lightglue.pipeline import SparsePipeline

    return SparsePipeline(build(config.components["extractor"]), build(config.components["matcher"]))


def _pipeline(name: str, extractor: str, variant: str, default: str, description: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        role="pipeline",
        description=description,
        load=None,
        presets={
            preset: {
                "components": {
                    "extractor": {"model": extractor, "preset": preset},
                    "matcher": {"model": "lightglue", "preset": preset, "variant": variant},
                }
            }
            for preset in PRECISIONS
        },
        default_preset=default,
        compose=_compose,
    )


RDD_LIGHTGLUE = _pipeline("rdd-lightglue", "rdd", "rdd", "fp8", "RDD detector/descriptor + LightGlue")
ALIKED_LIGHTGLUE = _pipeline("aliked-lightglue", "aliked", "aliked", "fp32", "ALIKED + LightGlue (cvg's ALIKED-trained weights)")
RACO_ALIKED_LIGHTGLUE = _pipeline(
    "raco-aliked-lightglue", "raco-aliked", "raco-aliked", "fp32",
    "RaCo keypoints + ALIKED descriptors + LightGlue (cvg's RaCo + ALIKED weights)",
)
SUPERPOINT_LIGHTGLUE = _pipeline("superpoint-lightglue", "superpoint", "superpoint", "fp32", "SuperPoint + LightGlue (cvg weights)")
DISK_LIGHTGLUE = _pipeline("disk-lightglue", "disk", "disk", "fp32", "DISK + LightGlue (cvg weights)")
XFEAT_LIGHTGLUE = _pipeline("xfeat-lightglue", "xfeat", "xfeat", "fp32", "XFeat + LighterGlue (verlab weights)")


def _raco_pipeline(descriptor: str, default: str) -> ModelSpec:
    return _pipeline(
        f"raco-{descriptor}-lightglue", f"raco-{descriptor}", descriptor, default,
        f"RaCo keypoints + {descriptor} descriptors + LightGlue with {descriptor}'s weights (trained on its own "
        f"keypoints, not RaCo's)",
    )


RACO_SUPERPOINT_LIGHTGLUE = _raco_pipeline("superpoint", "fp32")
RACO_DISK_LIGHTGLUE = _raco_pipeline("disk", "fp32")
RACO_XFEAT_LIGHTGLUE = _raco_pipeline("xfeat", "fp32")
RACO_RDD_LIGHTGLUE = _raco_pipeline("rdd", "fp8")
