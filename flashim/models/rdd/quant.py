"""RDD quantization presets and protected modules.

Presets:

- ``fp32``        : the checkpoint's own precision (matches the reference implementation).
- ``bf16``        : descriptor branch (ConvNeXt, projections, deformable transformer) in BF16; the keypoint
  detector, attention softmax and all sampling/position math stay FP32.
- ``fp8``         : ``bf16`` + the transformer's value/output projections and FFN linears as static MXFP8
  (E4M3 values, E8M0 block-32 scales), run by the fused deformable-encoder pass (default).
- ``fp8-full``    : ``fp8`` + the deformable-attention sampling-offset and attention-weight projections.
- ``fp8-backbone``: ``fp8`` + the ConvNeXt pointwise (MLP) linears. Slower than ``fp8``: these GEMMs are small-K
  and memory-bound, so separate quantize passes cost more than FP8 saves (kept for experiments).
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
CONVNEXT_LINEARS = ("backbone.stages.*.pwconv1", "backbone.stages.*.pwconv2")
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
    "fp8-full": Preset(
        "fp8-full",
        _config("fp8-full", *ENCODER_LINEARS, *DEFORMABLE_LINEARS),
        torch.bfloat16,
        description="fp8 + static MXFP8 sampling-offset and attention-weight projections",
    ),
    "fp8-backbone": Preset(
        "fp8-backbone",
        _config("fp8-backbone", *ENCODER_LINEARS, *CONVNEXT_LINEARS),
        torch.bfloat16,
        description="fp8 + static MXFP8 ConvNeXt MLP linears (slower; experiments only)",
    ),
}
