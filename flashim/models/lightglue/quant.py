"""LightGlue quantization presets and protected modules.

Presets:

- ``fp32`` : the checkpoint's precision (matches the reference implementation).
- ``bf16`` : transformer layers and assignment projections in BF16; positions, rotary math, the assignment
  softmaxes and the similarity matrix stay FP32.
- ``fp8``  : ``bf16`` + every transformer linear (QKV / output / FFN, self- and cross-attention) as static MXFP8
  (E4M3 values, E8M0 block-32 scales); attention itself stays BF16.
- ``int8`` : ``bf16`` + the same linears as static symmetric INT8 (per-channel weights, per-tensor activation scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flashim.quant.scheme import INT8_ACTIVATIONS, INT8_WEIGHTS, MXFP8, QuantConfig, QuantScheme

CALIBRATION_DIR = Path(__file__).parent / "calibration"
FP8_CALIBRATION = CALIBRATION_DIR / "lightglue_fp8_mega.safetensors"
INT8_CALIBRATION = CALIBRATION_DIR / "lightglue_int8_mega.safetensors"

# Never quantized: the Fourier position encoding, the assignment heads (similarity + matchability), token
# confidences, and the input projection (``nn.Identity`` for the RDD pairing; a small 128 -> 256 entry
# projection for the ALIKED pairing, same rationale as RDD's own protected `input_proj.*`).
PROTECTED = ("posenc.*", "log_assignment.*", "token_confidence.*", "input_proj.*")

TRANSFORMER_LINEARS = tuple(
    f"transformers.*.{leaf}" for leaf in ("Wqkv", "out_proj", "to_qk", "to_v", "to_out", "ffn.0", "ffn.3")
)


@dataclass(frozen=True)
class Preset:
    name: str
    config: QuantConfig | None
    compute_dtype: torch.dtype
    description: str = ""
    calibration: Path = FP8_CALIBRATION

    @property
    def int8(self) -> bool:
        return self.config is not None and any(
            s.weights.type == "int" for s in self.config.config_groups.values()
        )


PRESETS: dict[str, Preset] = {
    "fp32": Preset("fp32", None, torch.float32, description="checkpoint precision (reference numerics)"),
    "bf16": Preset("bf16", None, torch.bfloat16, description="BF16 transformer, FP32 positions/assignment"),
    "fp8": Preset(
        "fp8",
        QuantConfig({"linear": QuantScheme(TRANSFORMER_LINEARS, MXFP8, MXFP8)}, name="fp8"),
        torch.bfloat16,
        description="bf16 + static MXFP8 transformer linears",
    ),
    "int8": Preset(
        "int8",
        QuantConfig({"linear": QuantScheme(TRANSFORMER_LINEARS, INT8_WEIGHTS, INT8_ACTIVATIONS)}, name="int8"),
        torch.bfloat16,
        description="bf16 + static INT8 transformer linears",
        calibration=INT8_CALIBRATION,
    ),
}
