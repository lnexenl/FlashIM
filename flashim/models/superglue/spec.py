"""SuperGlue's :class:`~flashim.build.ModelSpec` (variants ``outdoor`` / ``indoor``) and the ``superpoint-superglue``
pipeline. See :mod:`flashim.models.superglue.quant` for measured quantization and speed evidence."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.superglue import quant, weights
from flashim.models.superglue.config import SuperGlueConfig
from flashim.models.superglue.model import MultiHeadedAttention, SuperGlue
from flashim.platforms import Feature


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


def _pack_qkv(model: SuperGlue, ctx: BuildContext) -> int:
    return model.pack_qkv()


def _low_precision_similarity(model: SuperGlue, ctx: BuildContext) -> str:
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.low_precision_similarity = True
    return "matmul_nt_fp32"


def _fused_int8_glue(model: SuperGlue, ctx: BuildContext) -> int | str:
    if ctx.config.precision.quant != "int8":
        return "skipped: not static int8"
    from flashim.models.superglue import int8_glue

    if (reason := int8_glue.unsupported(model)) is not None:
        return f"skipped: {reason}"
    if missing := int8_glue.probe(model):
        return f"skipped: no kernel on {ctx.platform.describe()} for {', '.join(missing)}"
    model.int8_glue = True
    return len(model.gnn.layers)


def _cuda_graph(model: SuperGlue, ctx: BuildContext) -> int | str:
    if ctx.config.engine.kind != "eager":
        return f"skipped: engine {ctx.config.engine.kind}"
    if not ctx.platform.supports(Feature.CUDA_GRAPHS):
        return f"skipped: no CUDA graphs on {ctx.platform.describe()}"
    model.graphs = int(ctx.option("cuda_graphs", 4))
    return model.graphs


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


PASSES = {
    "fold_batchnorm": Pass(_fold_batchnorm, stage="pre_quant", description="exact linear+BN1d fold"),
    "rotate_residual_stream": Pass(
        lambda model, ctx: quant.rotate_residual_stream(model),
        stage="pre_quant",
        exact=False,
        description="Hadamard rotations folded into GNN + final_proj weights (evens out 7-15x channel outliers)",
    ),
    "pack_qkv": Pass(_pack_qkv, stage="pre_quant", description="pack proj[0..2] into one qkv linear (3x fewer GEMMs)"),
    "low_precision_similarity": Pass(
        _low_precision_similarity,
        exact=False,
        description="similarity matrix via ops.matmul_nt_fp32 on BF16 tensor cores (FP32 accum and output)",
    ),
    "fused_int8_glue": Pass(
        _fused_int8_glue,
        description="static int8: [M+N, C] stream with multi-quantize, merge->mlp0 quantize, fused ReLU and residual",
    ),
    "cuda_graph": Pass(
        _cuda_graph,
        description="eager: kenc + GNN + similarity + Sinkhorn + mutual top-1 replayed from one CUDA graph per (M, N)",
    ),
}

_QUANT_CONFIGS = {name: p.config for name, p in quant.PRESETS.items() if p.config is not None}

SPEC = ModelSpec(
    name="superglue",
    role="matcher",
    description="SuperGlue sparse matcher: SparseFeatures x SparseFeatures -> matches (variant outdoor | indoor)",
    load=_load,
    presets={
        "fp32": {
            "optimize": {"passes": ["fold_batchnorm", "pack_qkv", "cuda_graph"]},
            "inference": {"sinkhorn_iterations": 100, "match_threshold": 0.2},
        },
        "bf16": {
            "extends": "fp32",
            "precision": {"compute_dtype": "bf16"},
            "optimize": {"passes+": ["low_precision_similarity"]},
        },
        "int8": {
            "extends": "fp32",
            "precision": {"compute_dtype": "bf16", "quant": "int8"},
            "optimize": {
                "passes+": [
                    "rotate_residual_stream",
                    "low_precision_similarity",
                    "fused_int8_glue",
                ]
            },
        },
    },
    default_preset="bf16",
    quant_configs=_QUANT_CONFIGS,
    calibrations={
        "int8": quant.CALIBRATION_DIR / f"superglue_{weights.DEFAULT_VARIANT}_int8_mega.safetensors",
        **{
            f"{variant}/int8": quant.CALIBRATION_DIR / f"superglue_{variant}_int8_mega.safetensors"
            for variant in weights.VARIANTS
        },
    },
    protected=quant.PROTECTED,
    passes=PASSES,
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


# Ship fp32 / bf16 / int8 on the pipeline (omitting fp8 so fp8 -> int8 via build.NEAREST): neither component benefits
# from MXFP8 (superglue has no fp8; superpoint fp8 is slower than int8 and drops AUC@5 by -0.0133 MegaDepth / -0.0144
# ScanNet due to 41.6x block-32 channel outliers in conv1b; see quant.py).
_PIPELINE_PRESETS = ("fp32", "bf16", "int8")

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
        for preset in _PIPELINE_PRESETS
    },
    default_preset="bf16",
    compose=_compose_pipeline,
    apply_inference=_apply_pipeline_inference,
    validate=_validate_pipeline,
)
