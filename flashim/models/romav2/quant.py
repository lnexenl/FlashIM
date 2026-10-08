"""RoMaV2 quantization presets and protected modules.

Public presets (what ``flashim.load("romav2", preset)`` is documented to take):

- ``bf16``    : no quantization.
- ``int8``    : static INT8 (per-channel weights, per-tensor percentile-calibrated activations, INT32
  accumulation) for every DINOv3 and matcher-ViT linear, the DPT head's four 1x1 input projections and the
  pointwise conv of the stride-4 / stride-2 refiner blocks, plus FlashIM refiner kernels: fused BF16 stride-1
  conv blocks, a fused refiner input (warp sampling, displacement embedding, local correlation in one pass) and
  fused output heads (FP32 warp/confidence convs + updates), one projection per feature map (RoMaV2 recomputes
  each for the second direction) and both directions refined as one batch. Not RoMaV2-exact (reduction order /
  rounding points).
- ``fp8``     : MXFP8 (E4M3 values + E8M0 block-32 scales) for every DINOv3 and matcher-ViT linear with static
  calibrated activation scales, fused MXFP8 stride-4 and stride-2 refiner blocks (dynamic MXFP8 pointwise) and
  the same FlashIM refiner kernels as ``int8``. Not RoMaV2-exact.

Oracle presets (``ORACLE_PRESETS``: kept for the parity tests and the comparison with RoMaV2's own FastFP8
runtime, not meant for use): ``fp8-p8`` is RoMaV2 "P8" (transformer linears only, SDPA attention and refiners
BF16), ``fp8-s4`` adds fused stride-4 refiner blocks, ``fp8-s42`` stride-4 and stride-2 (RoMaV2's
``s4_fused_s2_fused``); all bit-identical to the oracle.

Accuracy evidence for these configurations (RoMaV2 P8 reports, RTX 5070 Ti,
3 seeds, AUC@5 delta vs BF16): ``fp8-p8`` Mega -0.0025 / ScanNet -0.0011;
``fp8-s42`` Mega -0.0038 / ScanNet -0.0023 (Mega seed 17 alone: -0.0061).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flashim.quant.scheme import (
    INT8_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    MXFP8_DYNAMIC,
    QuantConfig,
    QuantScheme,
)

CALIBRATION_DIR = Path(__file__).parent / "calibration"
DEFAULT_CALIBRATION = CALIBRATION_DIR / "romav2_full_fp8_mega1500.safetensors"
INT8_CALIBRATION = CALIBRATION_DIR / "romav2_int8_mega1500.safetensors"
# INT8_CALIBRATION's entries + the VGG19-BN fine-feature convs (calibrate_romav2.py --merge).
INT8_VGG_CALIBRATION = CALIBRATION_DIR / "romav2_int8_vgg_mega1500.safetensors"

# Never quantized: coordinate-producing heads, geometry embeddings, the patch
# embedding and the refiner projections that feed FP32 sampling/correlation.
PROTECTED = (
    "f.patch_embed*",
    "refiners.*.proj",
    "refiners.*.disp_emb",
    "refiners.*.warp_head",
    "refiners.*.confidence_head",
    "matcher.head.scratch.output_conv2*",
)

_TRANSFORMER = QuantScheme(targets=("f.blocks.*", "matcher.mv_vit.*"), weights=MXFP8, input_activations=MXFP8)
_INT8_DPT = QuantScheme(
    targets=("matcher.head.projects.*",), weights=INT8_WEIGHTS, input_activations=INT8_ACTIVATIONS
)
_INT8_REFINER_S42 = QuantScheme(
    targets=("refiners.4.*.conv_pointwise", "refiners.2.*.conv_pointwise"),
    weights=INT8_WEIGHTS,
    input_activations=INT8_ACTIVATIONS,
)
# The VGG19-BN fine features' 3x3 convs (BatchNorm folded first: the ``fold_fine_features`` pass).
_INT8_FINE_FEATURES = QuantScheme(
    targets=("refiner_features.layers.*",), weights=INT8_WEIGHTS, input_activations=INT8_ACTIVATIONS
)
_INT8_TRANSFORMER = QuantScheme(
    targets=("f.blocks.*", "matcher.mv_vit.blocks.*"),
    weights=INT8_WEIGHTS,
    input_activations=INT8_ACTIVATIONS,
)


def _refiner_scheme(*strides: int) -> QuantScheme:
    return QuantScheme(
        targets=tuple(f"refiners.{s}.*.conv_pointwise" for s in strides),
        weights=MXFP8,
        input_activations=MXFP8_DYNAMIC,
    )


@dataclass(frozen=True)
class Preset:
    name: str
    config: QuantConfig | None
    fold_refiners: bool = False
    # VGG19-BN fine features with BatchNorm folded and channels-last weights (changes rounding).
    fold_fine_features: bool = False
    # DINOv3's RoPE as the fused ``ops.rope_qk`` kernel instead of the upstream math (one BF16 ulp on 24% of q / k).
    fused_rope: bool = False
    # Refiner stages whose BF16 conv blocks run as one fused kernel each (no quantization).
    fused_bf16_refiners: tuple[int, ...] = ()
    # Refiner glue as fused ops: block1 input (warp sampling, disp_emb, local corr) and
    # output heads (warp/confidence convs + updates).
    fused_refiner_io: bool = False
    # Refine both match directions as one batch with one projection per feature map (bit-exact).
    batched_directions: bool = False
    # Fuse each MXFP8 refiner block (depthwise + quantize + pointwise GEMM) into one kernel where supported.
    fused_mx_blocks: bool = False
    description: str = ""
    # Default static-activation calibration table for this preset.
    calibration: Path = DEFAULT_CALIBRATION


PRESETS: dict[str, Preset] = {
    "bf16": Preset("bf16", None, description="BF16 reference"),
    "int8": Preset(
        "int8",
        QuantConfig(
            {"transformer": _INT8_TRANSFORMER, "dpt": _INT8_DPT, "refiner": _INT8_REFINER_S42}, name="int8"
        ),
        fold_refiners=True,
        fused_bf16_refiners=(1,),
        fused_refiner_io=True,
        batched_directions=True,
        calibration=INT8_CALIBRATION,
        description="static INT8 transformer linears + DPT projections + stride-4/2 refiner blocks, FlashIM refiner "
        "kernels (fused BF16 stride-1 blocks, fused refiner input/heads, batched directions); not RoMaV2-exact",
    ),
    "fp8": Preset(
        "fp8",
        QuantConfig({"transformer": _TRANSFORMER, "refiner": _refiner_scheme(4, 2)}, name="fp8"),
        fold_refiners=True,
        fused_bf16_refiners=(1,),
        fused_refiner_io=True,
        batched_directions=True,
        fused_mx_blocks=True,
        description="static MXFP8 transformer linears + fused MXFP8 stride-4/2 refiner blocks, FlashIM refiner "
        "kernels (fused BF16 stride-1 blocks, fused refiner input); not RoMaV2-exact",
    ),
}

# Bit-identical to RoMaV2's FastFP8 runtime with the matching refiner variant (tests/parity/romav2).
ORACLE_PRESETS: dict[str, Preset] = {
    "fp8-p8": Preset(
        "fp8-p8",
        QuantConfig({"transformer": _TRANSFORMER}, name="fp8-p8"),
        description="RoMaV2 P8: static MXFP8 transformer linears, BF16 refiners",
    ),
    "fp8-s4": Preset(
        "fp8-s4",
        QuantConfig({"transformer": _TRANSFORMER, "refiner": _refiner_scheme(4)}, name="fp8-s4"),
        fold_refiners=True,
        description="fp8-p8 + fused MXFP8 stride-4 refiner blocks",
    ),
    "fp8-s42": Preset(
        "fp8-s42",
        QuantConfig({"transformer": _TRANSFORMER, "refiner": _refiner_scheme(4, 2)}, name="fp8-s42"),
        fold_refiners=True,
        description="fp8-p8 + fused MXFP8 stride-4 and stride-2 refiner blocks",
    ),
}
