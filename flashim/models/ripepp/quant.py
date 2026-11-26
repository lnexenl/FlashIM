"""RIPE++ quantization presets and calibration configuration.

Reuses RIPE's VGG19 + ConvRefiner quantization scheme:
- Quantized: VGG convs after the first (``net.encoder.layers.[1-9]*``) and ConvRefiner 1x1 convs
  (``block1.0``, ``block1.3``, ``hidden_blocks.*.3``; the 64/128-channel hidden blocks are replaced by fused BF16
  kernels before quantization so only the 256/512-channel ones are quantized).
- Protected: first conv (``net.encoder.layers.0``), heatmap ``out_conv``, and 960 -> 256 descriptor reduction
  (``conv_dim_reduction_coarse_desc``).
- ``fp8`` ignore: early VGG stem convs ``net.encoder.layers.{3, 7, 10}`` (64/128 ch) hit E4M3's 3-bit mantissa floor,
  amplified by the hypercolumn projection; keeping those three convs in BF16 leaves 32/35 convs in MXFP8 while
  preserving fidelity. INT8's 7-bit mantissa keeps them quantized.
- Variant presets: ``ripepp`` (default MegaDepth checkpoint) ships ``fp32``, ``bf16``, ``int8``, and ``fp8``.
  ``ripepp-aachen`` and ``ripepp-scared`` ship ``fp32`` and ``bf16`` only (resolving ``int8`` / ``fp8`` to ``bf16``
  via ``build.NEAREST``) because Tokyo 24/7 / SCARED calibration and accuracy-gate datasets are not available here.

Measured on RTX 5070 Ti (``sm_120``, ``eager``, 1200 px / 4096 kp on MegaDepth-1500, 640x480 / 4096 kp on
ScanNet-1500, PoseLib 1.0 px, seeds 17/29/42, paired bootstrap 95% CI vs ``bf16``, gate -0.005 AUC@5):
- ``ripepp bf16`` (33.7 ms extract): MegaDepth-1500 AUC@5 0.5877 (0.7107 / 0.8021), ScanNet-1500 0.1620 (0.3132 / 0.4689).
- ``ripepp int8`` (19.1 ms extract, 1.77x vs bf16): MegaDepth-1500 -0.0001 [-0.0057, +0.0055] (BORDERLINE),
  ScanNet-1500 +0.0010 [-0.0050, +0.0071] (PASS).
- ``ripepp fp8`` (27.6 ms extract, 1.22x vs bf16): MegaDepth-1500 -0.0047 [-0.0104, +0.0012] (BORDERLINE),
  ScanNet-1500 +0.0030 [-0.0026, +0.0088] (PASS).
- Fidelity vs ``fp32`` (MegaDepth 1200 px, 10 pairs/scene): ``bf16`` kp_overlap 0.9895 / desc_cos 0.9999 / match_agree 0.9519;
  ``int8`` 0.9494 / 0.9987 / 0.8131; ``fp8`` 0.9431 / 0.9992 / 0.8120.
"""

from __future__ import annotations

from flashim.models.ripe.spec import PROTECTED, QUANTIZED

# Early 64/128-channel VGG stem convs excluded from MXFP8 (same as RIPE).
FP8_IGNORE = ("net.encoder.layers.3", "net.encoder.layers.7", "net.encoder.layers.10")

__all__ = ["FP8_IGNORE", "PROTECTED", "QUANTIZED"]
