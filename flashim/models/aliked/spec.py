"""ALIKED's :class:`~flashim.build.ModelSpec` (presets, passes, engines); see ``quant.py`` for the precision story."""

from __future__ import annotations

import dataclasses

from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.aliked.config import VARIANTS
from flashim.models.aliked.model import ALIKED

DEFAULT_VARIANT = "aliked-n16"


def _load(config: RunConfig) -> ALIKED:
    variant = config.variant or DEFAULT_VARIANT
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    arch = dataclasses.replace(VARIANTS[variant], **config.arch) if config.arch else None
    return ALIKED.from_pretrained(arch, variant=variant, checkpoint=config.checkpoint, device=config.device)


def _fold_batchnorm(model: ALIKED, ctx: BuildContext) -> list[str]:
    from flashim.models.aliked.fuse import fold_batchnorm

    model.fused_batchnorm = fold_batchnorm(model)
    return model.fused_batchnorm


def _channels_last(model: ALIKED, ctx: BuildContext) -> None:
    model.set_channels_last(True)


def _deform_int8(model: ALIKED, ctx: BuildContext) -> list[str]:
    from flashim.models.aliked.deform_quant import quantize_deformable_convs
    from flashim.quant.calibration import Int8CalibrationTable

    table = ctx.option("deform_int8_calibration")
    if table is None:
        raise ValueError("pass 'deform_int8' needs optimize.options.deform_int8_calibration (a path or table)")
    if not isinstance(table, Int8CalibrationTable):
        table = Int8CalibrationTable.load(table)
    model.quantized_deform_convs = quantize_deformable_convs(model, table)
    return model.quantized_deform_convs


def _set_precision(model: nn.Module, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


PASSES = {
    "fold_batchnorm": Pass(_fold_batchnorm, stage="pre_quant", description="exact conv+BN fold (~30% faster)"),
    "channels_last": Pass(_channels_last, description="channels-last encoder (measured slower on sm_120)"),
    "deform_int8": Pass(
        _deform_int8, exact=False,
        description="INT8 deformable convs (block3/4); option deform_int8_calibration; wins at batch >= 4",
    ),
}

PRESETS = {
    "fp32": {"optimize": {"passes": ["fold_batchnorm"]}},
    "bf16": {"extends": "fp32", "precision": {"compute_dtype": "bf16"}},
}

SPEC = ModelSpec(
    name="aliked",
    role="extractor",
    description="ALIKED sparse detector/descriptor (deformable-conv encoder + SDDH); variant=aliked-n16 default",
    load=_load,
    presets=PRESETS,
    default_preset="fp32",
    passes=PASSES,
    engines={"eager": EngineSpec(), "compile": EngineSpec(("block1", "block2", "block3", "block4", "desc_head"))},
    set_precision=_set_precision,
)

