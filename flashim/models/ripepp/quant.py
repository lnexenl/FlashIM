"""RIPE++ quantization presets and calibration configuration.

Reuses RIPE's VGG19 + ConvRefiner quantization scheme:
- Quantized: VGG convs after the first (``net.encoder.layers.[1-9]*``) and ConvRefiner 1x1 convs
  (``block1.0``, ``block1.3``, ``hidden_blocks.*.3``; the 64/128-channel hidden blocks are replaced by fused BF16
  kernels before quantization so only the 256/512-channel ones are quantized).
- Protected: first conv (``net.encoder.layers.0``), heatmap ``out_conv``, and 960 -> 256 descriptor reduction
  (``conv_dim_reduction_coarse_desc``).
- ``fp8`` ignore: early 64/128-channel VGG stem convs ``net.encoder.layers.{3, 7, 10}`` and 64/128-channel decoder
  scale 1-2 ``block1.{0, 3}`` 1x1 convs (``net.decoder.layers.{1, 2}.*``; scales 1-2 hidden blocks are already fused
  BF16) hit E4M3's 3-bit mantissa floor. With only ``enc.{3, 7, 10}`` ignored (32/35 convs MXFP8, 27.6 ms extract),
  MegaDepth 20-pair fidelity vs ``bf16`` dropped to ``kp_overlap`` 0.9448 / ``match_agree`` 0.8215 and MegaDepth-1500
  AUC@5 dropped -0.0047 [-0.0104, +0.0012] (0.5831). Mixed encoder MXFP8 + decoder INT8 only reached ``kp`` 0.9584 /
  ``agree`` 0.8570 (27.2 ms). Excluding the 4 high-resolution decoder convs (``dec.{1, 2}``) leaves 28/35 convs in
  MXFP8, lifts 20-pair fidelity vs ``bf16`` to ``kp`` 0.9891 / ``cos`` 0.9994 / ``agree`` 0.9340 (``fp32``-relative
  50-pair fidelity: ``kp`` 0.9864 / ``cos`` 0.9992 / ``agree`` 0.9202), and brings MegaDepth-1500 to +0.0004 PASS at
  28.8 ms extract (vs 33.7 ms ``bf16``).
- ``int8`` ignore: ``net.decoder.layers.{1, 2}.*`` (the same 4 tiny 64/128-ch high-resolution decoder ``block1`` convs)
  are also kept BF16 in ``int8`` (leaving 31/35 convs in INT8 with all exact INT8 stem/refiner passes active, 20.4 ms
  extract = 1.65x vs ``bf16``). Static INT8 on those 4 convs degraded 3x3 softmax subpixel localization (``kp@0.5px``
  0.9030 -> 0.9524, mean displacement 0.067 -> 0.031 px), which dropped ``ripepp-lightglue`` MegaDepth-1500 by -0.0060
  even with an exact ``bf16`` matcher; excluding them lifts 50-pair ``fp32``-relative fidelity from ``kp`` 0.9494 /
  ``agree`` 0.8131 to ``0.9743 / 0.8728`` and restores ``ripepp-lightglue int8`` to -0.0011 [-0.0052, +0.0030].
- Variant presets: ``ripepp`` (default MegaDepth checkpoint) ships ``fp32``, ``bf16``, ``int8``, and ``fp8``.
  ``ripepp-aachen`` and ``ripepp-scared`` ship ``fp32`` and ``bf16`` only (resolving ``int8`` / ``fp8`` to ``bf16``
  via ``build.NEAREST``) because Tokyo 24/7 / SCARED calibration and accuracy-gate datasets are not available here.

Measured on RTX 5070 Ti (``sm_120``, ``eager``, 1200 px / 4096 kp on MegaDepth-1500, 640x480 / 4096 kp on
ScanNet-1500, PoseLib 1.0 px, seeds 17/29/42, paired bootstrap 95% CI vs ``bf16``, gate -0.005 AUC@5):
- ``ripepp bf16`` (33.7 ms extract): MegaDepth-1500 AUC@5 0.5877 (0.7107 / 0.8021), ScanNet-1500 0.1620 (0.3132 / 0.4689).
- ``ripepp int8`` (20.4 ms extract, 1.65x vs bf16): MegaDepth-1500 -0.0040 [-0.0094, +0.0015] (BORDERLINE),
  ScanNet-1500 +0.0020 [-0.0037, +0.0076] (PASS).
- ``ripepp fp8`` (28.8 ms extract, 1.17x vs bf16): MegaDepth-1500 +0.0004 [-0.0047, +0.0056] (PASS; up from -0.0047
  before excluding ``dec.{1, 2}``), ScanNet-1500 -0.0012 [-0.0065, +0.0040] (BORDERLINE, AUC@10 +0.0024).
- Fidelity vs ``fp32`` (MegaDepth 1200 px, 10 pairs/scene): ``bf16`` kp_overlap 0.9895 / desc_cos 0.9999 / match_agree 0.9519;
  ``int8`` 0.9743 / 0.9987 / 0.8728; ``fp8`` 0.9864 / 0.9992 / 0.9202.
"""

from __future__ import annotations

from flashim.models.ripe.spec import PROTECTED, QUANTIZED

# High-resolution 64/128-channel decoder scale 1-2 block1.{0,3} 1x1 convs excluded from INT8: static INT8 noise on
# these 4 convs perturbs RIPE++'s 3x3 softmax subpixel refinement (kp@0.5px 0.9030 -> 0.9524, mean subpixel error
# 0.067 -> 0.031 px), which costs -0.0060 [-0.0106, -0.0015] AUC@5 on MegaDepth-1500 when feeding LightGlue (proven via
# keypoint/descriptor swap: bf16_kpt+int8_desc -0.0011 PASS vs int8_kpt+bf16_desc -0.0072). Keeping these 4 convs BF16
# leaves 31/35 convs in INT8 and restores int8_ext + bf16_mat to -0.0011 [-0.0052, +0.0030] at <1 ms extract overhead.
INT8_IGNORE = (
    "net.decoder.layers.1.*",
    "net.decoder.layers.2.*",
)

# Early 64/128-channel VGG stem convs and 64/128-channel decoder scale 1-2 block1 1x1 convs excluded from MXFP8:
# E4M3's 3-bit mantissa floor drops keypoint overlap from 0.9891 to 0.9448 and match agreement from 0.9340 to 0.8215
# on those 4 high-resolution decoder convs at 0.0 ms latency penalty (27.6 vs 27.6 ms); 28/35 convs remain MXFP8.
FP8_IGNORE = (
    "net.encoder.layers.3",
    "net.encoder.layers.7",
    "net.encoder.layers.10",
    "net.decoder.layers.1.*",
    "net.decoder.layers.2.*",
)

__all__ = ["FP8_IGNORE", "INT8_IGNORE", "PROTECTED", "QUANTIZED"]
