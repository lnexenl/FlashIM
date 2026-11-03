"""``loma-extractor``: DaD keypoints + the variant's LoMa descriptors (variant b128 | b | l | g | r; presets ``fp32`` /
``bf16`` / ``int8`` / ``fp8``).

Built like DeDoDe-G's spec (DaD's detector, DeDoDe-B / -G descriptor networks), so its quantizable / protected globs and
fused passes carry over: ``int8`` quantizes the DINOv2 ViT with per-token dynamic INT8 (no calibration entries) and runs
every DeDoDe int8 pass (decoder fusions, encoder stem, ViT glue, patch-embedding GEMM, SageAttention on ViT blocks 2-23;
each skips itself where its kernel is missing, the ViT ones on B128's VGG-only descriptor). Extract from the JPEG at
1024 px, 4096 keypoints (eager, interleaved best of 3), int8 before -> after: L20 B / L / G / R 51.7 -> 33.6 ms (L's
pipeline: 34.6), B128 18.6 -> 14.3; H20 44.7 -> 31.6 (L: 32.2), B128 17.4 -> 13.2 (bf16 B: 57.8 / 49.5 ms). One calibration table per distinct descriptor weight set and
format (``scripts/calibrate_extractor.py --model loma-extractor --variant <v>``): B / L / G hold bit-equal descriptors
and share B's table, B128 and R have their own; the detector (``dad.pth``) is calibrated again on LoMa's 1024 px input
inside each table."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from flashim.config import RunConfig
from flashim.models.dedode.spec import (
    DECODER_FUSION_PASSES,
    DECODER_UPSAMPLE_PASS,
    PATCH_EMBED_PASS,
    PROTECTED,
    QUANTIZED,
    STEM_EXACT_PASSES,
    STEM_TC_PASSES,
    VIT_ATTENTION_PASSES,
    VIT_EPILOGUE_PASSES,
    VIT_LINEARS,
)
from flashim.models.loma.extractor import DESCRIPTORS, LoMaExtractor
from flashim.models.sparse_spec import extractor_spec

CALIBRATION_DIR = Path(__file__).parent / "calibration"
# variant -> its descriptor weight set's calibration tables (B / L / G: one descriptor, bit-equal in the checkpoints)
TABLES = {"b128": "b128", "b": "b", "l": "b", "g": "b", "r": "r"}


def _load(config: RunConfig) -> LoMaExtractor:
    return LoMaExtractor.from_variant(config.variant or "b", checkpoint=config.checkpoint, device=config.device)


def _validate(config: RunConfig) -> None:
    if (config.variant or "b") not in DESCRIPTORS:
        raise ValueError(f"loma-extractor: unknown variant {config.variant!r}; choose from {sorted(DESCRIPTORS)}")


def _apply_inference(model: LoMaExtractor, inference: Mapping) -> None:
    """``inference.descriptor_size``: ``[H, W]`` of the descriptor's resize (upstream 784 x 784)."""
    if "descriptor_size" in inference:
        model.descriptor_size = tuple(inference["descriptor_size"])


# DeDoDe-G's int8 passes (dedode.spec; each skips itself outside int8 / where its kernel is missing, the ViT ones on
# B128's VGG-only descriptor).
_PASSES = {
    **DECODER_UPSAMPLE_PASS, **DECODER_FUSION_PASSES, **STEM_EXACT_PASSES, **STEM_TC_PASSES, **VIT_EPILOGUE_PASSES,
    **PATCH_EMBED_PASS, **VIT_ATTENTION_PASSES,
}

_SPEC = extractor_spec(
    name="loma-extractor",
    description="LoMa's extractor: DaD keypoint detector (VGG11) + the variant's LoMa-trained DeDoDe descriptors "
    "(variant b128: DeDoDe-B 128-dim; b | l | g | r: DeDoDe-G)",
    load=_load,
    calibration_dir=CALIBRATION_DIR,
    quantized=QUANTIZED,
    protected=PROTECTED,
    int8_dynamic=VIT_LINEARS,  # per-token dynamic INT8 ViT, as dedode-g's int8 (see dedode.spec.VIT_IGNORE)
    compile_regions=("detector", "descriptor"),
    fused_bf16_blocks=("detector.decoder.", "descriptor.decoder."),  # as DaD's spec
    channels_last=True,
    extra_passes=_PASSES,
    structural_passes=tuple(_PASSES),
)
# Every pass the presets run, in order (``loma-l``'s int8 pipeline drops the numerics-changing stem passes: spec.py).
PASSES = tuple(_SPEC.presets["fp32"]["optimize"]["passes"])
SPEC = dataclasses.replace(
    _SPEC,
    calibrations={  # without a variant: B's
        **{quant: CALIBRATION_DIR / f"loma-extractor_b_{quant}.safetensors" for quant in _SPEC.quant_configs},
        **{f"{variant}/{quant}": CALIBRATION_DIR / f"loma-extractor_{table}_{quant}.safetensors"
           for variant, table in TABLES.items() for quant in _SPEC.quant_configs},
    },
    validate=_validate,
    apply_inference=_apply_inference,
)
