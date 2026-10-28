"""RDD quantization presets and protected modules.

Presets:

- ``fp32``        : the checkpoint's own precision (matches the reference implementation).
- ``bf16``        : descriptor branch (ConvNeXt, projections, deformable transformer) in BF16; the keypoint
  detector, attention softmax and all sampling/position math stay FP32.
- ``fp8``         : ``bf16`` + the transformer's value/output projections and FFN linears as static MXFP8
  (E4M3 values, E8M0 block-32 scales), run by the fused deformable-encoder pass (default).

Measured and dropped: the deformable-attention sampling-offset / attention-weight projections as MXFP8 as well (no
faster, they share one BF16 GEMM in the fused pass; MegaDepth-1500 AUC@5 0.5242 against 0.5418), and the ConvNeXt
pointwise (MLP) linears (slower: small-K, memory-bound GEMMs, so separate quantize passes cost more than FP8 saves).
The calibration table still covers the deformable projections (:data:`CALIBRATED`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flashim.quant.scheme import MXFP8, QuantConfig, QuantScheme

CALIBRATION_DIR = Path(__file__).parent / "calibration"
DEFAULT_CALIBRATION = CALIBRATION_DIR / "rdd_fp8_mega.safetensors"

# Never quantized: the keypoint detector (localization), the stem/downsampling and input projections
# (few channels, image-facing) and the matchability head.
PROTECTED = ("detector.*", "input_proj.*", "backbone.downsample_layers.*", "matchibility_head.*")

_LAYERS = "transformer.encoder.layers.*"
ENCODER_LINEARS = (
    f"{_LAYERS}.self_attn.value_proj",
    f"{_LAYERS}.self_attn.output_proj",
    f"{_LAYERS}.linear1",
    f"{_LAYERS}.linear2",
)
DEFORMABLE_LINEARS = (f"{_LAYERS}.self_attn.sampling_offsets", f"{_LAYERS}.self_attn.attention_weights")


def _config(name: str, *targets: str) -> QuantConfig:
    return QuantConfig({"linear": QuantScheme(targets=targets, weights=MXFP8, input_activations=MXFP8)}, name=name)


@dataclass(frozen=True)
class Preset:
    name: str
    config: QuantConfig | None
    compute_dtype: torch.dtype
    description: str = ""
    # Keypoint-detector trunk precision (the score head stays FP32).
    detector_dtype: torch.dtype = torch.float32
    calibration: Path = DEFAULT_CALIBRATION


PRESETS: dict[str, Preset] = {
    "fp32": Preset("fp32", None, torch.float32, description="checkpoint precision (reference numerics)"),
    "bf16": Preset("bf16", None, torch.bfloat16, description="BF16 descriptor branch, FP32 detector"),
    "fp8": Preset(
        "fp8",
        _config("fp8", *ENCODER_LINEARS),
        torch.bfloat16,
        description="bf16 + static MXFP8 transformer value/output/FFN linears (fused encoder)",
    ),
}

# The linears ``scripts/calibrate_rdd.py`` records (a superset of ``fp8``'s).
CALIBRATED = _config("calibrated", *ENCODER_LINEARS, *DEFORMABLE_LINEARS)
