"""LightGlue's spec and the sparse pipelines built around it (RDD / ALIKED / RaCo + ALIKED -> LightGlue)."""

from __future__ import annotations

import dataclasses

from torch import nn

from flashim.build import EngineSpec, ModelSpec, dtype
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
    calibrations={name: p.calibration for name, p in _QUANTIZED.items()},
    fp8_fallbacks={"fp8": "int8"},
    protected=quant.PROTECTED,
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
)


# -- sparse pipelines ---------------------------------------------------------------------------------------


def _compose(config: RunConfig, build) -> nn.Module:
    from flashim.models.lightglue.pipeline import SparsePipeline

    return SparsePipeline(build(config.components["extractor"]), build(config.components["matcher"]))


def _apply_inference(model: nn.Module, inference: dict) -> None:
    model.defaults = {**model.defaults, **inference}


def _matcher_precision(variant: str, preset: str) -> dict:
    """Quantized presets use the variant's own calibration table when one ships next to the default (RDD's) one."""
    table = quant.CALIBRATION_DIR / f"lightglue_{variant}_{preset}_mega.safetensors"
    return {"precision": {"calibration": str(table)}} if preset in _QUANTIZED and table.exists() else {}


def _pipeline(name: str, extractor: str, extractor_preset: str, variant: str, default: str, description: str):
    return ModelSpec(
        name=name,
        role="pipeline",
        description=description,
        load=None,
        presets={
            preset: {
                "components": {
                    "extractor": {"model": extractor, "preset": extractor_preset},
                    "matcher": {"model": "lightglue", "preset": preset, "variant": variant, **_matcher_precision(variant, preset)},
                }
            }
            for preset in ("fp32", "bf16", *_QUANTIZED)
        },
        default_preset=default,
        # the pipeline's preset, not only the matcher's: the INT8 preset carries the variant's INT8 calibration table
        fp8_fallbacks=SPEC.fp8_fallbacks,
        compose=_compose,
        apply_inference=_apply_inference,
    )


RDD_LIGHTGLUE = _pipeline(
    "rdd-lightglue", "rdd", "fp8", "rdd", "fp8",
    "RDD detector/descriptor + LightGlue (preset = LightGlue's; components.extractor.preset = RDD's, default fp8)",
)
ALIKED_LIGHTGLUE = _pipeline(
    "aliked-lightglue", "aliked", "fp32", "aliked", "fp32",
    "ALIKED + LightGlue (cvg's ALIKED-trained weights); components.extractor.preset = ALIKED's (default fp32)",
)
RACO_ALIKED_LIGHTGLUE = _pipeline(
    "raco-aliked-lightglue", "raco-aliked", "fp32", "raco-aliked", "fp32",
    "RaCo keypoints + ALIKED descriptors + LightGlue (cvg's RaCo + ALIKED weights); components.extractor.preset = "
    "RaCo/ALIKED's (default fp32)",
)
SUPERPOINT_LIGHTGLUE = _pipeline(
    "superpoint-lightglue", "superpoint", "fp32", "superpoint", "fp32",
    "SuperPoint + LightGlue (cvg weights); components.extractor.preset = SuperPoint's (default fp32)",
)
DISK_LIGHTGLUE = _pipeline(
    "disk-lightglue", "disk", "fp32", "disk", "fp32",
    "DISK + LightGlue (cvg weights); components.extractor.preset = DISK's (default fp32)",
)
XFEAT_LIGHTGLUE = _pipeline(
    "xfeat-lightglue", "xfeat", "fp32", "xfeat", "fp32",
    "XFeat + LighterGlue (verlab weights); components.extractor.preset = XFeat's (default fp32)",
)


def _raco_pipeline(descriptor: str, default: str) -> ModelSpec:
    return _pipeline(
        f"raco-{descriptor}-lightglue", f"raco-{descriptor}", default, descriptor, default,
        f"RaCo keypoints + {descriptor} descriptors + LightGlue with {descriptor}'s weights (trained on its own "
        f"keypoints, not RaCo's); components.extractor.preset = {descriptor}'s (default {default})",
    )


RACO_SUPERPOINT_LIGHTGLUE = _raco_pipeline("superpoint", "fp32")
RACO_DISK_LIGHTGLUE = _raco_pipeline("disk", "fp32")
RACO_XFEAT_LIGHTGLUE = _raco_pipeline("xfeat", "fp32")
RACO_RDD_LIGHTGLUE = _raco_pipeline("rdd", "fp8")


def legacy_overrides(
    *,
    variant: str = "rdd",
    engine: str = "eager",
    device=None,
    checkpoint=None,
    calibration=None,
    strict: bool = False,
    config: LightGlueConfig | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    attention: str = "sdpa",
    depth_confidence: float | None = None,
) -> dict:
    """``load_lightglue`` keyword arguments -> a config overlay."""
    from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable

    overlay: dict = {
        "variant": variant,
        "engine": {"kind": engine, "mode": compile_mode},
        "optimize": {"strict": strict, "attention": attention},
    }
    if device is not None:
        overlay["device"] = str(device)
    if checkpoint is not None:
        overlay["checkpoint"] = str(checkpoint)
    arch = dataclasses.asdict(config) if config is not None else {}
    if depth_confidence is not None:
        arch["depth_confidence"] = depth_confidence
    if arch:
        overlay["arch"] = arch
    if isinstance(calibration, (CalibrationTable, Int8CalibrationTable)):
        overlay["optimize"]["options"] = {"calibration_table": calibration}
    elif calibration is not None:
        overlay["precision"] = {"calibration": str(calibration)}
    return overlay
