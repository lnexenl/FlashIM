"""RIPE++ :class:`~flashim.build.ModelSpec` (``ripepp`` default MegaDepth, ``ripepp-aachen``, ``ripepp-scared``)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from flashim.build import ModelSpec
from flashim.config import RunConfig
from flashim.models.dedode.spec import (
    DECODER_FUSION_PASSES,
    DECODER_UPSAMPLE_PASS,
    STEM_EXACT_PASSES,
)
from flashim.models.ripepp import quant
from flashim.models.ripepp.model import RIPEPP
from flashim.models.sparse_spec import extractor_spec


def _loader(default_variant: str):
    def load(config: RunConfig) -> RIPEPP:
        variant = config.variant or default_variant
        return RIPEPP.from_pretrained(
            variant=variant,
            checkpoint=config.checkpoint,
            device=config.device,
            **config.arch,
        )

    return load


# Same bit-exact INT8 passes as RIPE (the 3 non-exact DeDoDe passes shift heatmap logits near the 0.5 threshold).
_PASSES = {
    **DECODER_UPSAMPLE_PASS,
    "int8_refiner_input": DECODER_FUSION_PASSES["int8_refiner_input"],
    **STEM_EXACT_PASSES,
}


def _spec(name: str, default_variant: str, description: str, *, quantized_presets: bool = True) -> ModelSpec:
    spec = extractor_spec(
        name=name,
        description=description,
        load=_loader(default_variant),
        calibration_dir=Path(__file__).parent / "calibration",
        quantized=quant.QUANTIZED,
        protected=quant.PROTECTED,
        int8_ignore=quant.INT8_IGNORE if quantized_presets else (),
        compile_regions=("net",),
        fused_bf16_blocks=("net.decoder.",),
        channels_last=True,
        extra_passes=_PASSES,
        structural_passes=tuple(_PASSES),
    )
    if not quantized_presets:
        presets = {k: {"variant": default_variant, **v} for k, v in spec.presets.items() if k in ("fp32", "bf16")}
        return dataclasses.replace(spec, presets=presets, quant_configs={}, calibrations={})
    return dataclasses.replace(
        spec,
        quant_configs={
            **spec.quant_configs,
            "fp8": dataclasses.replace(spec.quant_configs["fp8"], ignore=quant.FP8_IGNORE),
        },
    )


SPEC = _spec(
    "ripepp",
    "default",
    "RIPE++ detector/descriptor (VGG19 + ConvRefiner heatmap, subpixel NMS, hypercolumn 256-d descriptors, MNN)",
)
SPEC_AACHEN = _spec(
    "ripepp-aachen",
    "aachen",
    "RIPE++ detector/descriptor (MegaDepth + Tokyo 24/7 weights)",
    quantized_presets=False,
)
SPEC_SCARED = _spec(
    "ripepp-scared",
    "scared",
    "RIPE++ detector/descriptor (SCARED surgical video weights)",
    quantized_presets=False,
)
