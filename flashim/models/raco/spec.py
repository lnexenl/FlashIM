"""RaCo (detector + ranker) and RaCo + ALIKED (``raco-aliked``) specs."""

from __future__ import annotations

import dataclasses

from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.raco.config import RaCoConfig
from flashim.models.raco.model import RaCo


def _load(config: RunConfig) -> RaCo:
    arch = dataclasses.replace(RaCoConfig(), **config.arch) if config.arch else None
    return RaCo.from_pretrained(arch, checkpoint=config.checkpoint, device=config.device)


def _fold_batchnorm(model: RaCo, ctx: BuildContext) -> list[str]:
    from flashim.models.raco.fuse import fold_batchnorm

    model.fused_batchnorm = fold_batchnorm(model)
    return model.fused_batchnorm


def _fused_ranker(model: RaCo, ctx: BuildContext) -> str:
    model.use_fused_ranker(ctx.option("ranker_backend", "auto"))
    return model.ranker_backend


def _set_precision(model: nn.Module, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


SPEC = ModelSpec(
    name="raco",
    role="extractor",
    description="RaCo keypoint detector + ranker (cvg/RaCo)",
    load=_load,
    presets={
        "fp32": {"optimize": {"passes": ["fold_batchnorm"]}},
        "bf16": {"extends": "fp32", "precision": {"compute_dtype": "bf16"}, "optimize": {"passes+": ["fused_ranker"]}},
    },
    default_preset="fp32",
    passes={
        "fold_batchnorm": Pass(_fold_batchnorm, stage="pre_quant", description="exact conv+BN fold"),
        "fused_ranker": Pass(
            _fused_ranker, requires=lambda p: p.device_type == "cuda", exact=False,
            description="ranker residual blocks + head as fused ops (batch 1); option ranker_backend",
        ),
    },
    engines={
        "eager": EngineSpec(),
        "compile": EngineSpec(("block1", "block2", "block3", "block4", "ranker_head")),
    },
    set_precision=_set_precision,
)


def _compose(config: RunConfig, build) -> nn.Module:
    from flashim.models.raco.extractor import RaCoALIKED

    return RaCoALIKED(build(config.components["raco"]), build(config.components["aliked"]))


RACO_ALIKED = ModelSpec(
    name="raco-aliked",
    role="extractor",
    description="RaCo keypoints + ALIKED (aliked-n16) descriptors",
    load=None,
    presets={
        p: {"components": {"raco": {"model": "raco", "preset": p}, "aliked": {"model": "aliked", "preset": p}}}
        for p in ("fp32", "bf16")
    },
    default_preset="fp32",
    compose=_compose,
)
