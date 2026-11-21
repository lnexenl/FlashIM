"""Tiny-RoMa quantization configuration, outlier diagnosis, and measured preset evidence.

Outlier diagnosis on MegaDepth-1500 (``resize=1024``, RTX 5070 Ti / sm_120):
- ``coarse_matcher.0.layer.0`` (130 -> 256) concatenates 128 feature channels (max 16.25, median weight 0.069) with 2
  normalized warp coordinate channels in ``[-1, 1]`` (weight max 5.56, 80x larger). Naive per-tensor INT8 crushes the
  coordinate activations to ~8 levels and the feature weights to ~2 levels (20-pair ``med_px`` 3.81, ``@2px`` 0.199).
  SmoothQuant coordinate scaling (``coarse_coord_scale = 32.0`` folded into ``weight[:, -2:]``) restores full accuracy
  for INT8 (5-pair ``coarse_matcher.0`` alone: ``med_px`` 1.430 -> 0.131, ``@2px`` 0.637 -> 0.9989). For MXFP8 (E4M3,
  3 mantissa bits), power-of-two coordinate scaling cannot fix 3-bit mantissa truncation of ``[-1, 1]`` coordinates
  (``med_px`` 0.384, ``@2px`` 0.9714), so ``coarse_matcher.0`` stays BF16 in ``fp8``.
- ``fine_matcher.0.layer.0`` (50 -> 64) shows the same coordinate-channel mismatch (feature max 9.56 vs ``[-1, 1]``,
  weight max 4.94 vs median 0.53): naive INT8 ``med_px`` 3.134 -> SmoothQuant ``s=8.0`` ``med_px`` 0.352 (``@2px``
  0.260 -> 0.9970); MXFP8 ``med_px`` 1.660 (coordinates share block-32 with features).
- Speed (sacre_coeur 731x1024 vs 1024x768, eager, sm_120):
  - ``coarse_matcher`` (256-ch 3x3): BF16 0.745 ms -> INT8 0.303 ms (2.46x), FP8 0.311 ms (2.40x).
  - ``xfeat.0`` backbone (24-128 ch): BF16 convs 1.31 ms vs INT8 2.14 ms / FP8 2.26 ms (quantized convs slower on
    small channels < 128, matching FlashIM's known CNN result).
  - ``fine_matcher`` (64-ch 3x3): BF16 0.349 ms vs INT8 0.392 ms / FP8 0.384 ms (slower than BF16).

Presets therefore quantize the 256-channel ``coarse_matcher`` where INT8 / MXFP8 both win by ~2.4x and match BF16
fidelity, while keeping the small-channel CNNs (``xfeat.0``, ``fine_matcher``) in BF16 and the two 3-channel output
heads (``coarse_matcher.4``, ``fine_matcher.4``) in FP32:
- ``int8``: static INT8 on ``coarse_matcher.[0123].layer.0`` with SmoothQuant ``coarse_coord_scale=32.0``;
- ``fp8``: static MXFP8 on ``coarse_matcher.[123].layer.0`` (``coarse_matcher.0`` stays BF16 due to E4M3 3-bit
  mantissa on raw normalized coordinates).

Tried and rejected (20 MegaDepth-1500 pairs, ``resize=1024``, vs ``fp32``):
- ``bf16`` baseline: ``med_px=0.139``, ``mean_px=0.204``, ``@1px=0.9806``, ``@2px=0.9954``, ``cert_mae=0.0070``.
- ``int8`` all 22 convs (no smooth): ``med_px=3.808``, ``@2px=0.1992`` (broken by coord inputs + slower backbone).
- ``fp8`` all 22 convs: ``med_px=2.418``, ``@2px=0.4415`` (broken by ``fine_0`` / ``coarse_0`` + slower backbone).
- ``int8`` 20 convs (no ``coarse_0``/``fine_0``): ``med_px=0.214``, ``@2px=0.9874`` (slower backbone, +0.83 ms).
- ``fp8`` 20 convs (no ``coarse_0``/``fine_0``): ``med_px=0.356``, ``@2px=0.9610`` (slower backbone, +0.95 ms).
- ``int8`` ``coarse_123 + fine_123``: ``med_px=0.196``, ``@2px=0.9936`` (``fine_matcher`` slower than BF16).
- ``fp8`` ``coarse_123 + fine_123``: ``med_px=0.185``, ``@2px=0.9941`` (``fine_matcher`` slower than BF16).
- ``int8`` ``coarse_123`` only: ``med_px=0.158``, ``@2px=0.9939``.
- ``int8`` ``coarse_0123`` + SmoothQuant ``32x`` (kept as ``int8``): ``med_px=0.183``, ``mean_px=0.254``,
  ``@1px=0.9743``, ``@2px=0.9937``, ``@5px=0.9996``, ``cert_mae=0.0083`` (eager forward 5.30 ms vs BF16 5.74 ms).
- ``fp8`` ``coarse_123`` only (kept as ``fp8``): ``med_px=0.162``, ``mean_px=0.232``, ``@1px=0.9752``,
  ``@2px=0.9944``, ``@5px=0.9997``, ``cert_mae=0.0078`` (eager forward 5.39 ms vs BF16 5.74 ms).
"""

from __future__ import annotations

from pathlib import Path

from flashim.quant.scheme import INT8_ACTIVATIONS, INT8_WEIGHTS, MXFP8, QuantConfig, QuantScheme

CALIBRATION_DIR = Path(__file__).parent / "calibration"

COARSE_COORD_SCALE: float = 32.0

INT8_QUANTIZED = ("coarse_matcher.[0123].layer.0",)
FP8_QUANTIZED = ("coarse_matcher.[123].layer.0",)

PROTECTED = (
    "xfeat.0.block1.*",
    "xfeat.0.skip1.*",
    "xfeat.0.block_fusion.2",
    "coarse_matcher.4",
    "fine_matcher.4",
)

QUANT_CONFIGS: dict[str, QuantConfig] = {
    "int8": QuantConfig({"conv": QuantScheme(INT8_QUANTIZED, INT8_WEIGHTS, INT8_ACTIVATIONS)}, name="int8"),
    "fp8": QuantConfig({"conv": QuantScheme(FP8_QUANTIZED, MXFP8, MXFP8)}, name="fp8"),
}

CALIBRATIONS: dict[str, Path] = {
    fmt: CALIBRATION_DIR / f"tiny-roma_{fmt}.safetensors" for fmt in QUANT_CONFIGS
}
