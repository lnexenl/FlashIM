"""EfficientLoFTR quantization targets, protected modules and calibration tables.

Presets:

- ``fp32`` : checkpoint precision (reference numerics; RepVGG reparameterized + FPN BatchNorms folded).
- ``bf16`` : RepVGG backbone, coarse transformer and FPN in BF16; RoPE, coarse dual-softmax (``full``) and
  two-stage fine matching stay FP32. Default preset (upstream's recommended mixed-precision inference path).
- ``int8`` : ``bf16`` + the 48 coarse transformer linears (``q_proj``, ``k_proj``, ``v_proj``, ``merge``,
  ``mlp.0``, ``mlp.2``) as static symmetric INT8 (per-channel weights, calibrated per-tensor activation scales).
- ``fp8``  : ``bf16`` + the same 48 coarse transformer linears as static MXFP8 (E4M3 values, E8M0 block-32 scales).

Why ``backbone.*`` and ``fine_preprocess.*`` stay unquantized (measured on sm_120, MegaDepth-1500 pairs at 832 px):

- Quantizing the 20 sequential RepVGG 3x3 convolutions in ``backbone`` dropped 2 px match agreement vs FP32 from
  0.675 (BF16) to 0.288 (INT8) / 0.522 (MXFP8) (8 px agreement 0.911 -> 0.620 / 0.789; match ratio 1.004 -> 0.923 /
  0.964) and ran slower than cuDNN BF16 via im2col+GEMM.
- Keeping ``backbone.*`` and ``fine_preprocess.*`` in BF16 and quantizing only the 48 coarse transformer linears
  restores fidelity close to BF16 (``benchmarks/accuracy/fidelity.py``, 8 MegaDepth-1500 pairs at 832 px:
  BF16 2 px match_agree 0.7084 / ratio 1.0026; INT8 0.6896 / 1.0007; MXFP8 0.6908 / 1.0013) and runs slightly
  faster than BF16 in eager mode on sm_120 (640x832, best of 3: ``full`` fp32 34.90 / bf16 19.16 / int8 18.69 /
  fp8 18.74 ms/pair; ``opt`` fp32 34.69 / bf16 17.76 / int8 17.29 / fp8 17.33 ms/pair).
"""

from __future__ import annotations

from pathlib import Path

CALIBRATION_DIR = Path(__file__).parent / "calibration"

_COARSE_LINEARS = ("q_proj", "k_proj", "v_proj", "merge", "mlp.0", "mlp.2")

# Quantized: every coarse transformer linear (48 layers across 8 aggregated-attention blocks).
QUANTIZED = tuple(f"loftr_coarse.layers.*.{leaf}" for leaf in _COARSE_LINEARS)

# Protected: the RepVGG backbone (20 sequential 3x3 convs + 1-ch stem where static quantization error compounds),
# the depthwise query-aggregation convs, and the FPN decoder (``fine_preprocess``) feeding two-stage fine matching.
PROTECTED = ("backbone.*", "loftr_coarse.layers.*.aggregate", "fine_preprocess.*")
