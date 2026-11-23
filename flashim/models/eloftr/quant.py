"""EfficientLoFTR quantization targets, protected modules and calibration tables.

Presets:

- ``fp32`` : checkpoint precision (reference numerics; RepVGG reparameterized + FPN BatchNorms folded).
- ``bf16`` : RepVGG backbone + FPN (channels-last), fused NHWC coarse transformer (``pack_kv`` + ``fuse_coarse_mlp``
  + ``eloftr_agg_norm`` / ``eloftr_pool_norm`` / ``eloftr_upsample_add_leaky_relu`` / ``eloftr_post_norm_add`` + B=2
  self-attention batching) and BF16 tensor-core coarse similarity; RoPE, coarse dual-softmax (``full``) and two-stage
  fine matching stay FP32. Default preset.
- ``int8`` : ``bf16`` + split-identity RepVGG ``backbone.layer2`` / ``layer3`` (18 convs) and all 7 ``fine_preprocess``
  FPN convs (25 convs total) as static symmetric INT8 (per-channel weights, calibrated per-tensor activation scales).
- ``fp8``  : ``bf16`` + split-identity RepVGG ``backbone.layer2`` / ``layer3`` (18 convs) and 6 ``fine_preprocess``
  FPN convs (24 convs total; final ``layer1_outconv2.3`` kept in BF16) as static MXFP8.

Why RepVGG uses split-identity reparameterization and why ``loftr_coarse`` stays BF16 (measured on sm_120, 640x832):

- Folding ``rbr_identity`` BN into ``rbr_reparam.weight`` places values up to 69.17 on the center diagonal while
  ``rbr_dense + rbr_1x1`` stays below 0.81 (up to 7.9e6:1 per-row max/median spike), collapsing quantized 2 px
  agreement from 0.708 to 0.288 (INT8) / 0.522 (MXFP8). Keeping ``id_scale`` as an FP32 1D buffer added in the
  quantized conv epilogue removes the weight spike with zero extra kernel launch.
- ``backbone.layer0`` (1-ch stem) and ``backbone.layer1`` (2 early 64-ch blocks with 11.3x spatial outlier feeding
  ``fine_preprocess`` directly) stay in BF16; ``fine_preprocess.layer1_outconv2.3`` stays BF16 in ``fp8`` because
  E4M3's 3-bit mantissa degrades fine pixel argmax (2 px agreement 0.394 -> 0.683 when protected).
- After algebraic fusion, the entire ``loftr_coarse`` transformer takes only 1.97 ms in cuBLAS BF16; quantizing its
  small (256/512-dim) linears added +0.32 to +1.26 ms of launch/quantize overhead and reduced 2 px agreement by
  0.014-0.019 (INT8 8.21 ms / ag2 0.689 vs 9.47 ms / 0.675 with all coarse linears; FP8 9.41 ms / 0.683 vs 10.09 ms
  / 0.670), so keeping ``loftr_coarse`` in BF16 wins on both speed and accuracy.

Measured MegaDepth-1500 20-pair fidelity (``benchmarks/accuracy/fidelity.py``, 2 px match agreement vs FP32, PIL
decoder) and interleaved speedups on RTX 5070 Ti (sm_120, 640x832 eager vs commit ``2cf9bca``):

- ``bf16`` : 2 px agreement 0.7090, match ratio 1.0019; 1.52x faster (``opt``: 1.48x).
- ``int8`` : 2 px agreement 0.6796, match ratio 1.0007 (25 quantized convs); 2.31x faster vs old ``int8`` / 2.37x vs
  old ``bf16`` (``opt``: 2.30x / 2.36x).
- ``fp8``  : 2 px agreement 0.6838, match ratio 0.9976 (24 quantized convs); 2.05x faster vs old ``fp8`` / 2.10x vs
  old ``bf16`` (``opt``: 2.06x / 2.11x).
"""

from __future__ import annotations

from pathlib import Path

CALIBRATION_DIR = Path(__file__).parent / "calibration"

# Quantized: RepVGG layer2/layer3 (18 convs) and all 7 FPN convs (25 convs in int8; 24 in fp8 via FP8_IGNORE).
QUANTIZED = (
    "backbone.layer2.*.rbr_reparam",
    "backbone.layer3.*.rbr_reparam",
    "fine_preprocess.*",
)

# Protected: 1-ch stem, early layer1 (11.3x outlier feeding FPN), and the fused BF16 coarse transformer.
PROTECTED = (
    "backbone.layer0.*",
    "backbone.layer1.*",
    "loftr_coarse.*",
)

# Final 128->64 FPN conv feeding 8x8 fine pixel argmax: E4M3 3-bit mantissa is too coarse, INT8 7-bit keeps accuracy.
FP8_IGNORE = ("fine_preprocess.layer1_outconv2.3",)

INT8_DYNAMIC: tuple[str, ...] = ()
