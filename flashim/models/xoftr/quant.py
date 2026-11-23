"""XoFTR quantization targets, protected modules and measured scheme selection.

Presets (built by :mod:`flashim.models.xoftr.spec`):

- ``fp32`` : checkpoint precision (reference numerics; exact BatchNorm folding).
- ``bf16`` : backbone, coarse transformer and fine window-attention stages in BF16; input normalization, linear
  attention core (BF16 GEMMs via ``bf16_linear_attention``), coarse similarity / union-softmax matching, fine window
  softmaxes and sub-pixel coordinate math stay FP32.
- ``int8`` : ``bf16`` + static symmetric INT8 (per-output-channel weights, calibrated per-tensor activation scale) on
  the 13 ResNet_8_2 backbone convolutions from ``layer1.1`` through ``layer3_outconv`` (fused Triton implicit-GEMM with
  ReLU + residual epilogue).
- ``fp8`` : ``bf16`` + mixed-precision backbone quantization (3 MXFP8 convs on ``backbone.layer3.0.*`` + 8 static INT8
  convs on ``backbone.layer1.1.*``, ``backbone.layer2.[01].*``, ``backbone.layer3.1.conv1`` via the ``fp8_mixed_int8``
  post-quant pass; only the 2 final 1/8-res coarse-exit convs ``backbone.layer3.1.conv2`` and
  ``backbone.layer3_outconv`` stay NHWC BF16).

Measured scheme selection on MegaDepth-1500 (4 pairs/scene across 5 scenes = 20 pairs, 640 px,
``benchmarks/accuracy/fidelity.py`` match agreement within 2 px against ``fp32``, and eager latency on RTX 5070 Ti /
sm_120):

Phase 1 (before Triton kernel optimizations, ``sacre_coeur`` eager):

- ``bf16`` baseline: agreement 0.9263, match_ratio 1.0000, 18.15 ms/pair.
- All 87 conv+linear layers static INT8: 0.7858, 17.98 ms (coarse/fine linears 2.0x slower than BF16 due to small
  K=128..512 and separate quantize launches; ``backbone.layer1.0.conv1`` alone drops agreement to 0.8808 due to weight
  outliers max/med 627, per-in-ch max/med 37.0, and ``fine_process.*_attn_f`` drops to 0.8597).
- All 87 layers static MXFP8 (amax or p99.9 calibration): 0.6258 / 0.6252, 19.54 ms (same culprits: ``layer1`` 0.8169,
  ``fine_attn_f`` 0.7303).
- Alternative linear schemes tried before leaving coarse/fine linears in BF16:
  - SmoothQuant on all coarse+fine linears (alpha=0.5 / 0.75): agreement 0.8500 / 0.8572 (improves over 0.8229 static
    INT8 on those channel-outlier linears, ``ch_out`` up to 10.85 in ``loftr_coarse``, but still slower than BF16).
  - Dynamic per-token INT8 on coarse+fine linears: rescues ``fine_attn_f`` (0.8597 -> 0.8999) and ``fine_attn_m``
    (0.9015 -> 0.9177), all linears 0.8795; hybrid ``bb23 + dyn(coarse+fine_m)`` 0.8822 at 18.32 ms (still slower than
    BF16 18.15 ms because unfused dynamic INT8 linears at C=128..256 are launch/memory bound).
- Backbone-only quantization (``layer1.1`` .. ``layer3_outconv``, 13 convs; ``layer1.0`` kept BF16):
  - ``int8``: agreement 0.9027 (ratio 1.0004; cf. LoFTR ``int8`` 0.9042), 15.28 ms (1.19x vs ``bf16`` eager).
  - ``fp8`` (all-13 MXFP8): agreement 0.8705 (ratio 0.9967), 15.95 ms.
  (Quantizing ``layer1.0.conv2`` as well saves 0.50 ms in INT8 for -0.0085 agreement: 0.8942 / 14.78 ms.)

Phase 2 (after Triton kernel optimizations, 20 MegaDepth pairs, ``sacre_coeur`` / ``toronto`` eager on sm_120):

- Baselines:
  - ``bf16``: agreement 0.9310 (ratio 1.0006), 10.83 / 13.50 ms eager (10.01 / 12.61 ms ``compile``).
  - ``int8`` (all 13 INT8): agreement 0.9060 (ratio 1.0019), 8.15 / 11.02 ms eager (1.33x / 1.23x vs ``bf16``).
  - Old ``fp8`` (all 13 MXFP8): agreement 0.8736 (-0.0574 vs ``bf16``, ratio 0.9972), 8.68 / 11.46 ms eager.

Phase 3 (``fp8`` fidelity recovery — localization & rejected alternatives on the same 20 MegaDepth pairs):

- Root-cause attribution (all 13 convs, vs ``bf16`` 0.9310):
  - MXFP8 weights-only: 0.9047 (-0.0263); static MXFP8 acts-only: 0.8886 (-0.0424); dynamic per-pixel block-32 MXFP8
    acts-only: 0.8888 (-0.0423, +0.0002 vs static); dynamic acts + MXFP8 weights: 0.8701 (-0.0610, slightly below static
    0.8736). Weight relative L2 error is 0.0264..0.0266 in MXFP8 vs 0.0076..0.0145 in per-channel INT8 (2.0..3.5x
    higher). The drop is intrinsic E4M3 3-bit mantissa noise, not static activation outliers.
- Per-layer / per-stage localization (only that stage MXFP8 | leave-one-stage-out -> BF16 | swap -> INT8):
  - ``layer3_outconv``: 0.9105 (-0.0205) | 0.8859 (+0.0123) | 0.8834 (+0.0098) — worst single layer (0.026 ms BF16).
  - ``layer1.1``: 0.9144 (-0.0167) | 0.8832 (+0.0096) | 0.8755 (+0.0019) — fine 1/2 pyramid exit.
  - ``layer2.1``: 0.9159 (-0.0151) | 0.8819 (+0.0083) | 0.8772 (+0.0036) — medium 1/4 pyramid exit.
  - ``layer3.1``: 0.9198 (-0.0113) | 0.8785 (+0.0049) | 0.8770 (+0.0034) — coarse 1/8 pre-exit block.
  - ``layer2.0``: 0.9200 (-0.0111) | 0.8788 (+0.0051) | 0.8770 (+0.0034).
  - ``layer3.0``: 0.9236 (-0.0075) | 0.8775 (+0.0039) | 0.8734 (-0.0002) — virtually lossless in MXFP8 (-0.0004 vs INT8).
- Rejected MXFP8-only recovery schemes (all 13 convs unless noted):
  - Finer static calibration scaling (32 MegaDepth cal pairs): ``p99.9`` 0.8726, ``p99.5`` 0.8735, ``p99.0`` 0.8720,
    ``exp-1`` 0.8696 (no gain; confirms not outlier over-scaling).
  - Zero-cost folded BasicBlock transforms (``conv1 -> ReLU -> conv2``): channel-sort 0.8746, SmoothQuant alpha=0.5
    0.8721, sort+SmoothQuant 0.8733 (no gain).
  - Online block-32 Hadamard rotation upper bound: all 13 convs 0.8570 (-0.0166 worse than unrotated — smearing sparse
    ReLU maps exposes every entry to E4M3 3-bit rounding); ``layer3_outconv`` alone 0.9124 vs unrotated 0.9105.
  - Pure MXFP8 + selective BF16: 12 MX + bf16(``outconv``) 0.8859 (9.67 / 13.68 ms); 10 MX + bf16(3 exits) 0.8965
    (9.57 / 13.10 ms); 8 MX(``2.0,3.0,3.1``) + bf16(``1.1,2.1,outconv``) 0.9059 (10.20 / 13.49 ms);
    8 MX(``1.1,2.0,3.0``) + bf16(``2.1,3.1,outconv``) 0.9023 (9.40 / 12.79 ms);
    6 MX(``2.0,3.0``) + bf16(7 convs) 0.9155 (10.08 / 13.12 ms — loses almost all speedup over ``bf16``).
- Mixed MXFP8 + INT8 (+ minimal BF16) Pareto sweep (``sacre_coeur`` / ``toronto`` eager):
  - 8 MX(``2.0,3.0,3.1``) + 5 INT8: 0.8944 (8.75 / 13.06 ms); + bf16(``outconv``): 0.8949 (9.47 / 11.81 ms).
  - 6 MX(``2.0,3.0``) + 7 INT8: 0.9016; + bf16(``outconv``): 0.9044; + bf16(``3.1,outconv``): 0.9066 (8.62 / 11.57 ms);
    + bf16(``2.1,3.1,outconv``): 0.9102 (9.30 / 12.29 ms).
  - 5 MX(``3.0,3.1``) + 8 INT8: 0.8983 (9.05 / 11.74 ms); + bf16(``outconv``): 0.9008 (9.16 / 11.80 ms).
  - 3 MX(``3.0``) + 9 INT8 + bf16(``outconv``): 0.9094 (8.20 ms);
    3 MX(``3.0``) + 7 INT8 + bf16(``3.1,outconv``): 0.9110 (8.41 / 11.43 ms);
    3 MX(``3.0``) + 5 INT8 + bf16(``2.1,3.1,outconv``): 0.9159 (9.13 / 12.11 ms).
  - **Selected ``fp8``**: 3 MXFP8 (``layer3.0.*``) + 8 INT8 (``layer1.1.*``, ``layer2.[01].*``, ``layer3.1.conv1``) +
    2 BF16 (``layer3.1.conv2``, ``layer3_outconv``, 0.20 ms total in BF16 vs 0.10 ms quantized): agreement **0.9117**
    (-0.0193 vs ``bf16`` 0.9310, +0.0381 vs old all-MXFP8 0.8736, ratio 0.9976); interleaved best-of-3 eager latency
    with ``FLASHIM_AUTOTUNE_CACHE=0`` on sm_120: **9.23 / 11.80 ms** (1.18x / 1.14x vs ``bf16`` 10.86 / 13.44 ms;
    backbone 3.79 / 3.81 ms vs ``bf16`` 7.52 / 7.18 ms).
"""

from __future__ import annotations

QUANTIZED = (
    "backbone.layer1.1.conv[12]",
    "backbone.layer[23].[01].conv[12]",
    "backbone.layer[23].0.downsample.0",
    "backbone.layer3_outconv",
)

# Mixed-precision ``fp8`` preset: MXFP8 on the lossless downsampling stage (``layer3.0``), INT8 on the high-resolution
# compute-heavy blocks where E4M3's 3-bit mantissa degrades match agreement, leaving only the two tiny 1/8-res
# coarse-exit convolutions (``layer3.1.conv2``, ``layer3_outconv``) in NHWC BF16.
FP8_MXFP8 = (
    "backbone.layer3.0.conv[12]",
    "backbone.layer3.0.downsample.0",
)

FP8_INT8 = (
    "backbone.layer1.1.conv[12]",
    "backbone.layer2.[01].conv[12]",
    "backbone.layer2.0.downsample.0",
    "backbone.layer3.1.conv1",
)

PROTECTED = (
    "backbone.conv1",
    "backbone.layer1.0.*",
    "coarse_matching.final_proj",
    "fine_process.self_attn_[mf].pos_embed*",
    "fine_matching.fine_proj",
    "fine_matching.subpixel_mlp.*",
)
