"""LightGlue's spec and the sparse pipelines built around it (RDD / ALIKED / RaCo + ALIKED / ... -> LightGlue).

A pipeline's preset is every component's: each builds the nearest preset it has (``aliked-lightglue`` ``int8``: ALIKED
``bf16``, LightGlue ``int8``; ``model.build_report.fallbacks`` lists them); ``components.extractor.preset`` /
``components.matcher.preset`` set one component's."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from torch import nn

from flashim.build import PRECISIONS, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.lightglue import quant, weights
from flashim.models.lightglue.config import (
    ALIKED_CONFIG,
    DISK_CONFIG,
    RIPEPP_CONFIG,
    SUPERPOINT_CONFIG,
    XFEAT_CONFIG,
    LightGlueConfig,
)
from flashim.models.lightglue.model import CrossBlock, LightGlue, MatchAssignment, SelfBlock
from flashim.platforms import Feature

VARIANTS = {
    "rdd": LightGlueConfig(),
    "aliked": ALIKED_CONFIG,
    "raco-aliked": ALIKED_CONFIG,
    "superpoint": SUPERPOINT_CONFIG,
    "disk": DISK_CONFIG,
    "xfeat": XFEAT_CONFIG,
    "ripepp": RIPEPP_CONFIG,
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
    if variant == "ripepp":
        from flashim.models.ripepp.weights import load_lightglue_state_dict

        state = load_lightglue_state_dict(arch.num_heads, arch.num_layers, config.checkpoint)
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


def _low_precision_similarity(model: LightGlue, ctx) -> int:
    heads = [m for m in model.modules() if isinstance(m, MatchAssignment)]
    for head in heads:
        head.low_precision_similarity = True
    return len(heads)


def _cuda_graph(model: LightGlue, ctx) -> int | str:
    """Eager engine, no early stopping: replay ``best_matches`` from one CUDA graph per keypoint-count pair
    (bit-identical to eager; removes 59-76% CPU launch overhead at N=1024..2048)."""
    if ctx.config.engine.kind != "eager":
        return f"skipped: engine {ctx.config.engine.kind}"
    if model.config.depth_confidence > 0:
        return "skipped: early stopping (depth_confidence > 0)"
    if not ctx.platform.supports(Feature.CUDA_GRAPHS):
        return f"skipped: no CUDA graphs on {ctx.platform.describe()}"
    model.graphs = int(ctx.option("cuda_graphs", 4))
    return model.graphs


def _fused_bias(model: LightGlue, ctx) -> list[str] | str:
    """Fuse StaticMXFP8Linear biases into cuBLAS epilogue (fp8)."""
    if ctx.config.precision.quant != "fp8":
        return "skipped: not fp8"
    from flashim.quant.methods.linear import enable_fused_bias

    return enable_fused_bias(model)


def _fused_int8_glue(model: LightGlue, ctx) -> int | str:
    """int8: fuse the transformer layers' glue (:mod:`flashim.models.lightglue.int8_glue`, bit-identical) where the
    required ops have kernels for the platform."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim.models.lightglue import int8_glue

    if (reason := int8_glue.unsupported(model)) is not None:
        return f"skipped: {reason}"
    if missing := int8_glue.probe(model):
        return f"skipped: no kernel on {ctx.platform.describe()} for {', '.join(missing)}"
    model.int8_glue = True
    return len(model.transformers)


PASSES = {
    "low_precision_similarity": Pass(
        _low_precision_similarity, exact=False,
        description="assignment similarity as ops.matmul_nt_fp32 of final_proj's BF16 outputs instead of FP32 SGEMM",
    ),
    "cuda_graph": Pass(
        _cuda_graph,
        description="transformer + assignment head replayed from one CUDA graph per keypoint-count pair",
    ),
    "fused_bias": Pass(
        _fused_bias, exact=False,
        description="fp8: StaticMXFP8Linear biases fused into cuBLAS GEMM epilogue",
    ),
    "rotate_residual_stream": Pass(
        lambda model, ctx: quant.rotate_residual_stream(model), stage="pre_quant", exact=False,
        description="Hadamard rotations folded into the weights (int8: evens out the quantized inputs' channels)",
    ),
    "fused_int8_glue": Pass(
        _fused_int8_glue,
        description="int8: one residual stream for both images, quantize / RoPE / swap / residual fused into GEMMs",
    ),
}


_QUANTIZED = {name: p for name, p in quant.PRESETS.items() if p.config is not None}

SPEC = ModelSpec(
    name="lightglue",
    role="matcher",
    description="LightGlue sparse matcher: SparseFeatures x SparseFeatures -> matches (variant rdd | aliked | raco-aliked | superpoint | disk | xfeat | ripepp)",
    load=_load,
    presets={
        "fp32": {},
        "bf16": {
            "precision": {"compute_dtype": "bf16"},
            "optimize": {"passes": ["low_precision_similarity", "cuda_graph"]},
        },
        "fp8": {
            "extends": "bf16",
            "precision": {"quant": "fp8"},
            "optimize": {"passes+": ["fused_bias"]},
        },
        "int8": {
            "extends": "bf16",
            "precision": {"quant": "int8"},
            "optimize": {"passes+": ["rotate_residual_stream", "fused_int8_glue"]},
        },
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
    passes=PASSES,
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
)


# -- sparse pipelines ---------------------------------------------------------------------------------------


def _compose(config: RunConfig, build) -> nn.Module:
    from flashim.models.lightglue.pipeline import SparsePipeline

    return SparsePipeline(build(config.components["extractor"]), build(config.components["matcher"]))


def _pipeline(
    name: str,
    extractor: str,
    variant: str,
    default: str,
    description: str,
    *,
    presets: tuple[str, ...] = PRECISIONS,
    extractor_presets: Mapping[str, str] | None = None,
    matcher_presets: Mapping[str, str] | None = None,
) -> ModelSpec:
    ext_presets = extractor_presets or {}
    mat_presets = matcher_presets or {}
    return ModelSpec(
        name=name,
        role="pipeline",
        description=description,
        load=None,
        presets={
            preset: {
                "components": {
                    "extractor": {"model": extractor, "preset": ext_presets.get(preset, preset)},
                    "matcher": {"model": "lightglue", "preset": mat_presets.get(preset, preset), "variant": variant},
                }
            }
            for preset in presets
        },
        default_preset=default,
        compose=_compose,
        compile_failure_components=("extractor",),
    )


RDD_LIGHTGLUE = _pipeline("rdd-lightglue", "rdd", "rdd", "fp8", "RDD detector/descriptor + LightGlue")
# MXFP8 fails or borders the -0.005 AUC@5 gate on ALIKED, SuperPoint, DISK and XFeat LightGlue (see quant.py for
# measured evidence): omitting unsupported presets lets build.NEAREST resolve ``fp8`` to each pairing's nearest
# passing config.
ALIKED_LIGHTGLUE = _pipeline(
    "aliked-lightglue", "aliked", "aliked", "fp32", "ALIKED + LightGlue (cvg's ALIKED-trained weights)",
    presets=("fp32", "bf16", "int8"),
)
RACO_ALIKED_LIGHTGLUE = _pipeline(
    "raco-aliked-lightglue", "raco-aliked", "raco-aliked", "fp32",
    "RaCo keypoints + ALIKED descriptors + LightGlue (cvg's RaCo + ALIKED weights)",
)
# Quantized SuperPoint convs are slower than BF16 and shift descriptors off LightGlue's manifold (-0.041 fp8 / -0.009
# int8 on MegaDepth-300); keeping the extractor in BF16 lets the rotated INT8 matcher pass both MegaDepth and ScanNet.
SUPERPOINT_LIGHTGLUE = _pipeline(
    "superpoint-lightglue", "superpoint", "superpoint", "fp32", "SuperPoint + LightGlue (cvg weights)",
    presets=("fp32", "bf16", "int8"),
    extractor_presets={"int8": "bf16"},
)
# DISK + LightGlue MXFP8 drops -0.0047 on dev ScanNet-1500 (-0.0042 / AUC@10 -0.0063 even with bf16 extractor) and
# -0.0061 on integrated MegaDepth-1500 while running 1.5-2.4x slower than int8/bf16 on ScanNet; rotated INT8 passes
# both datasets (+0.0039 mega / +0.0010 scannet) and is faster.
DISK_LIGHTGLUE = _pipeline(
    "disk-lightglue", "disk", "disk", "fp32", "DISK + LightGlue (cvg weights)",
    presets=("fp32", "bf16", "int8"),
)
# XFeat's dim=96 1-head LighterGlue fails MXFP8 (-0.138 mega / -0.070 scannet) and exceeds -0.005 under rotated INT8
# (-0.0066 at p=0.9999, -0.0049 at p=1.0), while BF16 is faster on ScanNet (16.3 vs 19.5s) and exact (+0.0000).
XFEAT_LIGHTGLUE = _pipeline(
    "xfeat-lightglue", "xfeat", "xfeat", "fp32", "XFeat + LighterGlue (verlab weights)",
    presets=("fp32", "bf16"),
)
# RIPE++ + LightGlue:
# - MXFP8 collapses on MegaDepth-200 (-0.0538 AUC@5, match_ratio 0.676, 66.2 ms/pair); omitted (fp8 -> int8).
# - Quantizing RIPE++'s RL-trained LightGlue matcher drops MegaDepth-1500 AUC@5 under every scheme tested (-0.0034 to
#   -0.0081 static rotated, -0.0020 to -0.0077 dynamic rotated, -0.0041 to -0.0046 protecting L4-5) because late-layer
#   similarity/matchability logit scale explodes (sim MAE 36.85 vs RDD 0.58; matchability MAE 4.73, bias -0.63).
# - Keeping the matcher in bf16 (12.2 ms, 15% of bf16 pair time) while running the RIPE++ extractor in int8 (31/35 convs
#   INT8 with dec.{1,2} kept BF16 for subpixel accuracy) recovers MegaDepth-1500 to -0.0011 [-0.0052, +0.0030] while
#   delivering a ~1.5x pair speedup vs bf16 (see quant.py).
RIPEPP_LIGHTGLUE = _pipeline(
    "ripepp-lightglue", "ripepp", "ripepp", "bf16",
    "RIPE++ detector/descriptor + weakly-supervised LightGlue (RIPE++ RL weights)",
    presets=("fp32", "bf16", "int8"),
    matcher_presets={"int8": "bf16"},
)


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
