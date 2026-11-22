"""XoFTR quantization targets, protected modules and measured scheme selection.

Presets (built by :mod:`flashim.models.xoftr.spec` via :func:`~flashim.models.sparse_spec.extractor_spec`):

- ``fp32`` : checkpoint precision (reference numerics; exact BatchNorm folding).
- ``bf16`` : backbone, coarse transformer and fine window-attention stages in BF16; input normalization, linear
  attention core (BF16 GEMMs via ``bf16_linear_attention``), coarse similarity / union-softmax matching, fine window
  softmaxes and sub-pixel coordinate math stay FP32.
- ``int8`` : ``bf16`` + static symmetric INT8 (per-output-channel weights, calibrated per-tensor activation scale) on
  the ResNet_8_2 backbone convolutions from ``layer1.1`` through ``layer3_outconv`` (fused Triton implicit-GEMM with
  ReLU + residual epilogue).
- ``fp8`` : ``bf16`` + static MXFP8 (E4M3 + E8M0 block-32 scales) on the same backbone convolutions.

Measured scheme selection on MegaDepth-1500 (4 pairs/scene = 16 pairs, 640 px, ``benchmarks/accuracy/fidelity.py``
match agreement within 2 px against ``fp32``, and eager latency on RTX 5070 Ti / sm_120 on ``sacre_coeur``):

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
- Selected backbone-only quantization (``layer1.1`` .. ``layer3_outconv``, 13 convs; ``layer1.0`` kept BF16):
  - ``int8``: agreement 0.9027 (ratio 1.0004; cf. LoFTR ``int8`` 0.9042), 15.28 ms (1.19x vs ``bf16`` eager).
  - ``fp8`` : agreement 0.8705 (ratio 0.9967; cf. LoFTR ``fp8`` 0.7782), 15.95 ms (1.14x vs ``bf16`` eager).
  (Quantizing ``layer1.0.conv2`` as well saves 0.50 ms in INT8 for -0.0085 agreement: 0.8942 / 14.78 ms.)
- After Phase 2 Triton kernel optimizations (interleaved best-of-2 on RTX 5070 Ti / sm_120, 640 px, eager):
  - ``bf16``: agreement 0.9310 (ratio 1.0006), 10.83 ms (``sacre_coeur``) / 13.50 ms (``toronto``); ``compile`` 10.01 / 12.61 ms.
  - ``int8``: agreement 0.9060 (ratio 1.0019), 8.15 ms (``sacre_coeur``) / 11.02 ms (``toronto``) — 1.33x / 1.23x vs ``bf16``.
  - ``fp8`` : agreement 0.8736 (ratio 0.9972), 8.68 ms (``sacre_coeur``) / 11.46 ms (``toronto``) — 1.25x / 1.18x vs ``bf16``.
"""

from __future__ import annotations

QUANTIZED = (
    "backbone.layer1.1.conv[12]",
    "backbone.layer[23].[01].conv[12]",
    "backbone.layer[23].0.downsample.0",
    "backbone.layer3_outconv",
)

PROTECTED = (
    "backbone.conv1",
    "backbone.layer1.0.*",
    "coarse_matching.final_proj",
    "fine_process.self_attn_[mf].pos_embed*",
    "fine_matching.fine_proj",
    "fine_matching.subpixel_mlp.*",
)
