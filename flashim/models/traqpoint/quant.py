"""TraqPoint quantization presets and protected modules.

Presets:

- ``fp32`` : the checkpoint's own precision (matches the reference implementation).
- ``bf16`` : descriptor branch (DINOv3 ConvNeXt-B, projections, deformable transformer) in BF16; the keypoint
  detector, attention softmax and all sampling / position math stay FP32.
- ``int8`` : ``bf16`` + backbone ConvNeXt stages 2-3 pointwise MLP linears (60 linears, 30/36 blocks) as per-token
  dynamic INT8 (``DynamicInt8Linear``) + deformable-transformer value / output / FFN linears (16) as static symmetric
  INT8.
- ``fp8``  : ``bf16`` + deformable-transformer value / output / FFN linears (16) as static MXFP8 (fused deformable
  encoder) + backbone ConvNeXt stages 2-3 pointwise MLP linears (60) as per-token dynamic INT8.

Quantization diagnosis (20 MegaDepth pairs at 1600 px / 4096 kpts, fidelity vs FP32, and eager ``extract`` latency at
1200 px on RTX 5070 Ti / sm_120):

- ``fp32``: 76.7 ms
- ``bf16``: 31.3 ms, desc_cos 0.9999, 2 px match agreement 0.9906
- Static per-tensor INT8 on the DINOv3 ConvNeXt-B backbone loses fidelity due to per-spatial-token activation outliers
  at high resolution (backbone-only static INT8: 27.2 ms, desc_cos 0.9974, match agreement 0.9599; all-88 static INT8:
  26.0 ms, 0.9594; tx MXFP8 + all-backbone static INT8: 25.0 ms, 0.9557; tx MXFP8 + stages 2-3 static INT8: 26.0 ms,
  0.9629).
- Per-token dynamic INT8 on the backbone eliminates the spatial outlier clipping (backbone-only dynamic INT8: 28.5 ms,
  desc_cos 0.9991, match agreement 0.9760). Keeping memory-bound early stages 0-1 in BF16 and quantizing stages 2-3
  improves both fidelity and efficiency over all-stage dynamic INT8 or MXFP8:
  - ``int8`` (stages 2-3 dynamic INT8 + tx static INT8): 27.7 ms, desc_cos 0.9989, match agreement 0.9789 (-0.0117 vs
    bf16); vs all-stage dynamic INT8 + tx static INT8: 27.4 ms, 0.9726.
  - ``fp8`` (stages 2-3 dynamic INT8 + tx MXFP8 fused encoder): 26.7 ms, desc_cos 0.9983, match agreement 0.9757
    (-0.0149 vs bf16); faster and higher fidelity than stages 2-3 MXFP8 + tx MXFP8 (27.2 ms, 0.9726) and all-88 MXFP8
    (27.7 ms, 0.9654).
- Transformer-only quantization (backbone BF16): tx MXFP8 29.2 ms, match agreement 0.9815; tx static INT8 30.3 ms,
  0.9835.

Accuracy evidence (MegaDepth-1500 at 1600 px / 4096 kpts and ScanNet-1500 at 640x480 / 4096 kpts, PoseLib LO-RANSAC
1.0 px, seeds 17/29/42, eager, RTX 5070 Ti; ``benchmarks/accuracy/b_models_matrix.sh traqpoint``), AUC@5 and paired
bootstrap 95% CI vs ``bf16`` (``B=10000``):

- MegaDepth-1500: ``fp32`` 0.6499 (+0.0006 [-0.0023, +0.0035]), ``bf16`` 0.6494, ``int8`` 0.6515 (+0.0022 [-0.0008,
  +0.0052], PASS), ``fp8`` 0.6521 (+0.0028 [-0.0005, +0.0060], PASS).
- ScanNet-1500: ``fp32`` 0.2108 (-0.0007 [-0.0049, +0.0036]), ``bf16`` 0.2114, ``int8`` 0.2143 (+0.0028 [-0.0012,
  +0.0070], PASS), ``fp8`` 0.2094 (-0.0020 [-0.0061, +0.0020], BORDERLINE).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flashim.quant.scheme import (
    INT8_ACTIVATIONS,
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)

CALIBRATION_DIR = Path(__file__).parent / "calibration"
FP8_CALIBRATION = CALIBRATION_DIR / "traqpoint_fp8_mega.safetensors"
INT8_CALIBRATION = CALIBRATION_DIR / "traqpoint_int8_mega.safetensors"

# Never quantized: keypoint detector (localization), stem/downsampling and input projections, matchability head.
PROTECTED = ("detector.*", "input_proj.*", "backbone.downsample_layers.*", "matchibility_head.*")

BACKBONE_LINEARS = tuple(f"backbone.stages.*.*.{leaf}" for leaf in ("pwconv1", "pwconv2"))
BACKBONE_S23_LINEARS = tuple(f"backbone.stages.{s}.*.{leaf}" for s in (2, 3) for leaf in ("pwconv1", "pwconv2"))
_LAYERS = "transformer.encoder.layers.*"
ENCODER_LINEARS = (
    f"{_LAYERS}.self_attn.value_proj",
    f"{_LAYERS}.self_attn.output_proj",
    f"{_LAYERS}.linear1",
    f"{_LAYERS}.linear2",
)


@dataclass(frozen=True)
class Preset:
    name: str
    config: QuantConfig | None
    compute_dtype: torch.dtype
    description: str = ""
    calibration: Path = FP8_CALIBRATION


PRESETS: dict[str, Preset] = {
    "fp32": Preset("fp32", None, torch.float32, description="checkpoint precision (reference numerics)"),
    "bf16": Preset("bf16", None, torch.bfloat16, description="BF16 descriptor branch, FP32 detector"),
    "int8": Preset(
        "int8",
        QuantConfig(
            {
                "transformer": QuantScheme(ENCODER_LINEARS, INT8_WEIGHTS, INT8_ACTIVATIONS),
                "backbone": QuantScheme(BACKBONE_S23_LINEARS, INT8_WEIGHTS, INT8_TOKEN_ACTIVATIONS),
            },
            name="int8",
        ),
        torch.bfloat16,
        description="bf16 + dynamic INT8 backbone stages 2-3 MLP + static INT8 deformable transformer",
        calibration=INT8_CALIBRATION,
    ),
    "fp8": Preset(
        "fp8",
        QuantConfig(
            {
                "transformer": QuantScheme(ENCODER_LINEARS, MXFP8, MXFP8),
                "backbone": QuantScheme(BACKBONE_S23_LINEARS, INT8_WEIGHTS, INT8_TOKEN_ACTIVATIONS),
            },
            name="fp8",
        ),
        torch.bfloat16,
        description="bf16 + static MXFP8 deformable transformer (fused encoder) + dynamic INT8 backbone stages 2-3 MLP",
        calibration=FP8_CALIBRATION,
    ),
}

# Superset of all linears calibrated by ``scripts/calibrate_traqpoint.py``.
CALIBRATED_FP8 = QuantConfig(
    {"linear": QuantScheme(BACKBONE_LINEARS + ENCODER_LINEARS, MXFP8, MXFP8)},
    name="calibrated_fp8",
)
CALIBRATED_INT8 = QuantConfig(
    {"linear": QuantScheme(BACKBONE_LINEARS + ENCODER_LINEARS, INT8_WEIGHTS, INT8_ACTIVATIONS)},
    name="calibrated_int8",
)
