"""RoMaV2 quantization presets and protected modules.

Presets (all MXFP8, E4M3 values + E8M0 block-32 scales):

- ``bf16``    : no quantization.
- ``int8``    : every DINOv3 and matcher-ViT linear as symmetric INT8 (per-channel weights, static
  per-tensor percentile-calibrated activations, INT32 accumulation); attention and refiners BF16.
- ``int8-dpt``: ``int8`` + the DPT head's four 1x1 input projections.
- ``int8-full``: ``int8-dpt`` + the pointwise conv of every refiner block (depthwise + ReLU produce the
  INT8 activation directly); RoMaV2's ``recipes/int8_full.yaml`` (its ``layer*_rn`` rules match 3x3 convs
  and are inert there).
- ``fp8-p8``  : RoMaV2 "P8" -- every DINOv3 and matcher-ViT linear with static
  calibrated activation scales; SDPA attention stays BF16; refiners BF16.
- ``fp8-s4``  : ``fp8-p8`` + stride-4 refiner blocks fused (dynamic MXFP8 pointwise).
- ``fp8``     : ``fp8-p8`` + stride-4 and stride-2 refiner blocks fused (default).
- ``fp8-fast``: ``fp8`` + FlashIM refiner kernels: fused BF16 stride-1 conv blocks, a fused
  refiner input (warp sampling, displacement embedding, local correlation in one pass) and
  fused output heads (FP32 warp/confidence convs + updates), one projection per feature map
  (RoMaV2 recomputes each for the second direction) and both directions refined as one batch.
  Numerics differ from RoMaV2 by reduction order / rounding points only; pending AUC.

Accuracy evidence for these configurations (RoMaV2 P8 reports, RTX 5070 Ti,
3 seeds, AUC@5 delta vs BF16): ``fp8-p8`` Mega -0.0025 / ScanNet -0.0011;
``fp8`` Mega -0.0038 / ScanNet -0.0023 (Mega seed 17 alone: -0.0061).
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
_INT8_REFINER = QuantScheme(
    targets=("refiners.*.block1.conv_pointwise", "refiners.*.hidden_blocks.*.conv_pointwise"),
    weights=INT8_WEIGHTS,
    input_activations=INT8_ACTIVATIONS,
)
_INT8_REFINER_S42 = QuantScheme(
    targets=("refiners.4.*.conv_pointwise", "refiners.2.*.conv_pointwise"),
    weights=INT8_WEIGHTS,
    input_activations=INT8_ACTIVATIONS,
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
        QuantConfig({"transformer": _INT8_TRANSFORMER}, name="int8"),
        calibration=INT8_CALIBRATION,
        description="static INT8 transformer linears (per-channel weights, per-tensor activations), BF16 refiners",
    ),
    "int8-dpt": Preset(
        "int8-dpt",
        QuantConfig({"transformer": _INT8_TRANSFORMER, "dpt": _INT8_DPT}, name="int8-dpt"),
        calibration=INT8_CALIBRATION,
        description="int8 + INT8 DPT input projections (1x1 convs)",
    ),
    "int8-full": Preset(
        "int8-full",
        QuantConfig(
            {"transformer": _INT8_TRANSFORMER, "dpt": _INT8_DPT, "refiner": _INT8_REFINER}, name="int8-full"
        ),
        fold_refiners=True,
        calibration=INT8_CALIBRATION,
        description="int8-dpt + static INT8 pointwise convs in every refiner block (RoMaV2 int8_full recipe)",
    ),
    "int8-fast": Preset(
        "int8-fast",
        QuantConfig(
            {"transformer": _INT8_TRANSFORMER, "dpt": _INT8_DPT, "refiner": _INT8_REFINER_S42}, name="int8-fast"
        ),
        fold_refiners=True,
        fused_bf16_refiners=(1,),
        fused_refiner_io=True,
        batched_directions=True,
        calibration=INT8_CALIBRATION,
        description="int8-full with INT8 refiner blocks at strides 4/2 only + FlashIM refiner kernels "
        "(fused BF16 stride-1 blocks, fused refiner input/heads, batched directions); not RoMaV2-exact",
    ),
    "fp8-p8": Preset(
        "fp8-p8",
        QuantConfig({"transformer": _TRANSFORMER}, name="fp8-p8"),
        description="static MXFP8 transformer linears, BF16 refiners",
    ),
    "fp8-s4": Preset(
        "fp8-s4",
        QuantConfig({"transformer": _TRANSFORMER, "refiner": _refiner_scheme(4)}, name="fp8-s4"),
        fold_refiners=True,
        description="fp8-p8 + fused MXFP8 stride-4 refiner blocks",
    ),
    "fp8": Preset(
        "fp8",
        QuantConfig({"transformer": _TRANSFORMER, "refiner": _refiner_scheme(4, 2)}, name="fp8"),
        fold_refiners=True,
        description="fp8-p8 + fused MXFP8 stride-4 and stride-2 refiner blocks",
    ),
    "fp8-fast": Preset(
        "fp8-fast",
        QuantConfig({"transformer": _TRANSFORMER, "refiner": _refiner_scheme(4, 2)}, name="fp8-fast"),
        fold_refiners=True,
        fused_bf16_refiners=(1,),
        fused_refiner_io=True,
        batched_directions=True,
        fused_mx_blocks=True,
        description="fp8 + FlashIM refiner kernels (fused BF16 stride-1 blocks, fused refiner input); "
        "not RoMaV2-exact",
    ),
}
