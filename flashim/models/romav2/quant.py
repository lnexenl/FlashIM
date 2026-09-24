"""RoMaV2 quantization presets and protected modules.

Presets (all MXFP8, E4M3 values + E8M0 block-32 scales):

- ``bf16``    : no quantization.
- ``fp8-p8``  : RoMaV2 "P8" -- every DINOv3 and matcher-ViT linear with static
  calibrated activation scales; SDPA attention stays BF16; refiners BF16.
- ``fp8-s4``  : ``fp8-p8`` + stride-4 refiner blocks fused (dynamic MXFP8 pointwise).
- ``fp8``     : ``fp8-p8`` + stride-4 and stride-2 refiner blocks fused (default).
- ``fp8-fast``: ``fp8`` + FlashIM refiner kernels: fused BF16 stride-1 conv blocks and a fused
  refiner input (warp sampling, displacement embedding, local correlation in one pass).
  Numerics differ from RoMaV2 by reduction order / rounding points only; pending AUC.

Accuracy evidence for these configurations (RoMaV2 P8 reports, RTX 5070 Ti,
3 seeds, AUC@5 delta vs BF16): ``fp8-p8`` Mega -0.0025 / ScanNet -0.0011;
``fp8`` Mega -0.0038 / ScanNet -0.0023 (Mega seed 17 alone: -0.0061).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flashim.quant.scheme import MXFP8, MXFP8_DYNAMIC, QuantConfig, QuantScheme

CALIBRATION_DIR = Path(__file__).parent / "calibration"
DEFAULT_CALIBRATION = CALIBRATION_DIR / "romav2_full_fp8_mega1500.safetensors"

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
    # Build refiner block1 inputs (proj, warp, disp_emb, local corr) with one fused op.
    fused_refiner_input: bool = False
    description: str = ""


PRESETS: dict[str, Preset] = {
    "bf16": Preset("bf16", None, description="BF16 reference"),
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
        fused_refiner_input=True,
        description="fp8 + FlashIM refiner kernels (fused BF16 stride-1 blocks, fused refiner input); "
        "not RoMaV2-exact",
    ),
}
