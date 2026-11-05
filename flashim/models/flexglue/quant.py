"""FlexGlue quantization presets and protected modules.

Presets:

- ``fp32`` : the checkpoint's precision (matches the reference implementation).
- ``bf16`` : descriptor and transformer in BF16; the correspondence heads, search map, positions, entmax readout
  and mutual-NN stay FP32.
- ``fp8``  : ``bf16`` + every transformer linear (QKV / output / FFN, self- and cross-attention; 81) as static
  MXFP8 (E4M3 values, E8M0 block-32 scales) and the pointwise MLP linears of every ConvNeXt block of the
  descriptor's backbone (72) as static INT8 (every FP8 GPU has INT8 tensor cores; the descriptor loses less in
  INT8, the transformer less in MXFP8); attention itself stays BF16 (``optimize.attention`` swaps it).
- ``int8`` : ``bf16`` + the same 153 linears all as static symmetric INT8 (per-channel weights, per-tensor
  activation scale): the preset of GPUs without FP8.

The backbone's convolutions, the descriptor's stride-8 head, the input projection, the Fourier position encoding and
the correspondence heads (to_search / matchability / covisibility / offset MLPs) stay unquantized: the heads feed
the coordinate-sensitive entmax readout.

Accuracy evidence (MegaDepth-1500, dense ``match()`` with 5000 KDE-sampled matches at 512x512, poselib 1.0px, RANSAC
seeds 17/29/42, eager, RTX 5070 Ti; ``benchmarks/accuracy/flexglue_matrix.sh``), AUC@5 and the paired-bootstrap
difference to bf16 (95% CI):

- fp32 0.6389 (-0.0008 [-0.0021, +0.0006]), bf16 0.6397, fp8 0.6369 (-0.0028 [-0.0059, +0.0004]),
  int8 0.6370 (-0.0027 [-0.0057, +0.0002]); with the attention of each preset's GPUs: fp8 +
  ``optimize.attention=sage-fp8-fast`` (FP8 PV, sm_89+) 0.6371 (-0.0026 [-0.0061, +0.0008]), int8 +
  ``optimize.attention=sage`` (INT8 QK + FP16 PV, sm_80+) 0.6380 (-0.0017 [-0.0047, +0.0014]).
- fp8 with an MXFP8 descriptor (all 153 linears MXFP8): 0.6357 (-0.0040 [-0.0073, -0.0009]), + sage-fp8-fast
  0.6359 (-0.0038 [-0.0076, -0.0002]), both measurably below bf16; the INT8 descriptor is +0.0012 over it
  ([-0.0024, +0.0051] paired, both attentions) at the same speed (compiled ``match_keypoints`` 27.5 ms either way).
- The descriptor linears cost MXFP8 0.0031 / INT8 0.0010-0.0019 AUC@5 (transformer-only: fp8 0.6388 (-0.0009),
  int8 0.6380 (-0.0017)) for ~1.6 ms of a ~20 ms compiled fp8 + sage-fp8-fast ``match_keypoints`` (descriptor
  6.5 -> 4.7 ms); MXFP8 stages 2-3 only: 0.6374 (-0.0023) for ~1.2 ms. Kept for the speed, inside the 0.005 gate.
- Attention (transformer-only fp8): ``sage`` 0.6372 (-0.0025 [-0.0050, -0.0002]), ``sage-fp8`` 0.6375
  (-0.0022 [-0.0051, +0.0005]), ``sage-fp8-fast`` 0.6380 (-0.0016 [-0.0046, +0.0009]): measurably below, so
  SageAttention stays opt-in. SageAttention3's NVFP4 attention: 0.5475 (-0.092), unusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flashim.quant.scheme import INT8_ACTIVATIONS, INT8_WEIGHTS, MXFP8, QuantConfig, QuantScheme

CALIBRATION_DIR = Path(__file__).parent / "calibration"
FP8_CALIBRATION = CALIBRATION_DIR / "flexglue_fp8_mega.safetensors"
INT8_CALIBRATION = CALIBRATION_DIR / "flexglue_int8_mega.safetensors"

# Never quantized: the descriptor's stride-8 head (LinearConvHead, the fused descriptor_head), the Fourier position
# encoding, the input projection and the correspondence heads (their readout is coordinate-sensitive, like
# LightGlue's protected assignment heads). The backbone's convolutions are not linears and stay BF16.
PROTECTED = ("descriptor.head.*", "descriptor.descriptor_head", "posenc_wr", "input_proj", "heads.*")

# The ConvNeXt blocks' pointwise MLP (72 linears, 83.7M of the descriptor's 89.6M parameters), every stage.
DESCRIPTOR_LINEARS = tuple(f"descriptor.backbone.stages.*.*.{leaf}" for leaf in ("pwconv1", "pwconv2"))

TRANSFORMER_LINEARS = tuple(
    f"transformers.*.{leaf}"
    for leaf in ("Wqkv", "out_proj", "to_qk", "to_v", "to_out", "ffn.0", "ffn.3")
)


@dataclass(frozen=True)
class Preset:
    name: str
    config: QuantConfig | None
    compute_dtype: torch.dtype
    description: str = ""
    calibration: Path | tuple[Path, ...] = FP8_CALIBRATION


PRESETS: dict[str, Preset] = {
    "fp32": Preset("fp32", None, torch.float32, description="checkpoint precision (reference numerics)"),
    "bf16": Preset("bf16", None, torch.bfloat16, description="BF16 descriptor/transformer, FP32 heads/readout"),
    "fp8": Preset(
        "fp8",
        QuantConfig(
            {
                "transformer": QuantScheme(TRANSFORMER_LINEARS, MXFP8, MXFP8),
                "descriptor": QuantScheme(DESCRIPTOR_LINEARS, INT8_WEIGHTS, INT8_ACTIVATIONS),
            },
            name="fp8",
        ),
        torch.bfloat16,
        description="bf16 + static MXFP8 transformer linears + static INT8 descriptor MLP",
        calibration=(FP8_CALIBRATION, INT8_CALIBRATION),
    ),
    "int8": Preset(
        "int8",
        QuantConfig(
            {"linear": QuantScheme(TRANSFORMER_LINEARS + DESCRIPTOR_LINEARS, INT8_WEIGHTS, INT8_ACTIVATIONS)}, name="int8"
        ),
        torch.bfloat16,
        description="bf16 + static INT8 transformer linears and descriptor MLP",
        calibration=INT8_CALIBRATION,
    ),
}
