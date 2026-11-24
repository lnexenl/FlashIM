"""RoMa (v1) quantization schemes, protected modules, and presets.

Presets (measured on RTX 5070 Ti, sm_120, eager, default 560x560 -> 864x864, 10 MegaDepth-1500 pairs across all 5
scenes, confident pixels ``cert > 0.05`` vs ``fp32``; baseline before kernel optimization: fp32 483.9 ms, bf16
240.3 ms, int8 220.8 ms, fp8 220.9 ms):

- ``fp32``: checkpoint precision, no BatchNorm fold (bit-identical to upstream ``romatch`` with ``local_corr:reference``).
  Forward 446.7 ms / full ``match`` 456.3 ms (Triton ``local_corr`` extended to C <= 512, R <= 7).
- ``bf16``: BF16 VGG19-BN, DINOv2 ViT-L/14, coarse TransformerDecoder, fine projections (scales 8/4/2/1), and
  exact-padded refiner conv stacks (1377/1137/569/144/24 -> 1408/1152/576/192/32; ``ops.depthwise5x5_relu`` +
  ``F.linear`` on scales 16/8/4/2, fused 32-ch ``ops.depthwise5x5_relu_pointwise`` on scale 1, ``fused_refiner_io`` +
  last-dim coarse softmax); scale-16 projection, GP, ``disp_emb`` and ``out_conv`` stay FP32.
  Forward 124.1 ms / ``match`` 130.9 ms (1.94x vs baseline); median warp error 0.020 px (p90 0.088 px, 99.67% < 2 px,
  cert MAE 0.0023). Indoor (4 pairs): 124.0 ms, median 0.023 px, 99.90% < 2 px.
- ``int8``: dynamic per-token INT8 on all 116 transformer linears (96 DINOv2 ViT-L/14 + 20 coarse TransformerDecoder,
  fused via ``DynamicInt8TransformerBlock`` + ``fold_layer_scale`` + ``layernorm_quantize_int8_rows`` on sm_120),
  static per-tensor INT8 on all 36 padded refiner pointwise 1x1s at scales 16/8/4/2 (TileLang TMA
  ``depthwise5x5_relu_int8`` + Triton ``int8_linear``) and 11 VGG19-BN convs after ``layers.0`` (``fused_pool_epilogue``
  + ``fused_first_conv_epilogue``), plus fused 32-ch BF16 scale-1 refiners and ``fused_refiner_io`` (47 static layers
  calibrated on 16 MegaDepth pairs).
  Forward 66.9 ms / ``match`` 73.5 ms (3.59x vs BF16 baseline); median warp error 0.054 px (p90 0.192 px,
  99.38% < 2 px, cert MAE 0.0058). Indoor (4 pairs): 66.9 ms, median 0.077 px, 99.19% < 2 px.
- ``fp8``: static MXFP8 on all 116 transformer linears (cuBLAS block-scaled MXFP8 on Blackwell, ``fold_layer_scale`` +
  ``fused_block_epilogues``, calibrated on 16 MegaDepth pairs) + dynamic MXFP8 on all 36 padded refiner blocks at
  scales 16/8/4/2 (TileLang TMA ``depthwise5x5_relu_mxfp8`` + cuBLAS ``mxfp8_scaled_mm``) + fused 32-ch BF16 scale-1
  refiners and ``fused_refiner_io``. VGG19-BN stays BF16 (measured below).
  Forward 86.7 ms / ``match`` 93.0 ms (2.77x vs BF16 baseline); median warp error 0.052 px (p90 0.230 px,
  99.31% < 2 px, cert MAE 0.0062). Indoor (4 pairs): 86.8 ms, median 0.070 px, 99.53% < 2 px.

Rejected / diagnostic quantization schemes (10 hard ``calibration_local`` pairs; bf16 ref 99.25% < 2 px, p90 0.216 px):

- Static per-tensor INT8 on coarse ``embedding_decoder``: 98.09% < 2 px, p90 0.447 px (vs dynamic per-token INT8:
  98.61% < 2 px, p90 0.364 px; MegaDepth-1500 99.24% -> 99.38% < 2 px) -> switched to dynamic per-token INT8.
- SmoothQuant folded into ``SeparableConvBlock`` (alpha=0.5 / 0.75): 92.9 ms, 97.47% / 97.91% < 2 px, p90 0.689 /
  0.665 px (worse: shifts channel scale variance into per-output-channel weights across 1408 input channels).
- SageAttention (``sage`` / ``sage-fp8`` / ``sage-fp8-warp`` on DINOv2 blocks 2-23, sm_120): 92.5-94.2 ms vs 94.4 ms
  (flash SDPA is already 5.8 ms) while dropping < 2 px to 98.01-98.21% and p90 to 0.446-0.479 px -> rejected.
- Static MXFP8 on VGG19-BN in ``fp8``: all layers (3..36) 101.9->81.3 ms but drops < 2 px 98.63% -> 98.04% (p90 0.538),
  scales 2+4+8 (7..36) 81.3 ms / 97.73% (p90 0.661), scales 4+8 (14..36) 85.4 ms / 98.15% (p90 0.491), scale 8 only
  (27..36) 90.8 ms / 98.25% (p90 0.469) -> VGG19-BN stays BF16 in ``fp8`` (3-bit E4M3 mantissa degrades fine features).
- BF16 on ``decoder.proj["16"]``: hurts GP Cholesky in ``int8`` (p90 0.372 -> 0.428 px, < 2 px 98.57% -> 98.23%);
  casting only ``proj["8","4","2","1"]`` keeps GP FP32 and saves the same ~3.0 ms across ``bf16``/``int8``/``fp8``.
- ``minima-roma`` (modality-invariant weights, 10 MegaDepth-1500 pairs + MINIMA RGB-Depth demo, sm_120):
  ``bf16`` 135.7 ms, 99.83% < 2 px (med 0.023 px; demo 88.63%), ``fp8`` 97.6 ms, 99.23% < 2 px (med 0.058 px;
  demo 87.94%), ``int8`` 75.8 ms calibrated on 16 RGB MegaDepth pairs + 8 synthetic modality-augmented calibration
  pairs (inverted / autocontrast / edge / equalized) achieves 99.13% < 2 px on MegaDepth (med 0.057 px, p90 0.237 px)
  and improves cross-modal RGB-Depth demo fidelity from 78.19% (med 0.473 px, RGB-only calibration) to 80.85% < 2 px
  (med 0.423 px) at identical speed. Keeping VGG19-BN in BF16 under ``minima-roma`` ``int8`` was rejected (demo 76.89%).
"""

from __future__ import annotations

from pathlib import Path

from flashim.quant.scheme import (
    INT8_ACTIVATIONS,
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    MXFP8_DYNAMIC,
    QuantConfig,
    QuantScheme,
)

CALIBRATION_DIR = Path(__file__).parent / "calibration"

# Padded refiner channel counts (exact zero-padding after BatchNorm fold):
# scales 16/8/4/2 are multiples of 64 (TileLang TMA depthwise producers + cuBLAS MXFP8 / INT8 / BF16 GEMMs),
# scale 1 (24 -> 32) hits the fused 32-channel BF16 depthwise+pointwise kernel.
REFINER_PADDED_CHANNELS: dict[str, int] = {"16": 1408, "8": 1152, "4": 576, "2": 192, "1": 32}

# Never quantized: first 3-channel VGG conv, DINOv2 patch embedding, GP coordinate embedding,
# coarse classifier head, refiner projections (feed FP32 sampling / local_corr), displacement embeddings,
# and coordinate/certainty output heads.
PROTECTED = (
    "encoder.cnn.layers.0",
    "encoder.dinov2.patch_embed.*",
    "decoder.gps.*",
    "decoder.embedding_decoder.to_out",
    "decoder.proj.*",
    "decoder.conv_refiner.*.disp_emb",
    "decoder.conv_refiner.*.out_conv",
)

DINOV2_LINEARS = (
    "encoder.dinov2.blocks.*.attn.qkv",
    "encoder.dinov2.blocks.*.attn.proj",
    "encoder.dinov2.blocks.*.mlp.fc1",
    "encoder.dinov2.blocks.*.mlp.fc2",
)
DECODER_LINEARS = (
    "decoder.embedding_decoder.blocks.*.attn.qkv",
    "decoder.embedding_decoder.blocks.*.attn.proj",
    "decoder.embedding_decoder.blocks.*.mlp.fc1",
    "decoder.embedding_decoder.blocks.*.mlp.fc2",
)
REFINER_POINTWISE = (
    "decoder.conv_refiner.16.block1.3",
    "decoder.conv_refiner.16.hidden_blocks.*.3",
    "decoder.conv_refiner.8.block1.3",
    "decoder.conv_refiner.8.hidden_blocks.*.3",
    "decoder.conv_refiner.4.block1.3",
    "decoder.conv_refiner.4.hidden_blocks.*.3",
    "decoder.conv_refiner.2.block1.3",
    "decoder.conv_refiner.2.hidden_blocks.*.3",
)
VGG_CONVS = ("encoder.cnn.layers.[1-9]*",)

# Static MXFP8 on all 116 transformer linears (cuBLAS block-scaled MXFP8 on Blackwell) +
# dynamic MXFP8 on all 36 refiner blocks at scales 16/8/4/2 (TileLang TMA depthwise producer -> cuBLAS scaled_mm).
_FP8_TRANSFORMER = QuantScheme(
    targets=(*DINOV2_LINEARS, *DECODER_LINEARS),
    weights=MXFP8,
    input_activations=MXFP8,
)
_FP8_REFINERS = QuantScheme(
    targets=REFINER_POINTWISE,
    weights=MXFP8,
    input_activations=MXFP8_DYNAMIC,
)

# DINOv2 ViT-L/14 and coarse TransformerDecoder both use dynamic per-token INT8 (fixes both DINOv2 token
# outliers and static-INT8 coarse decoder drift; fused via DynamicInt8TransformerBlock). Refiner pointwise
# 1x1s (scales 16/8/4/2) and VGG19-BN (after layers.0) use static per-tensor INT8.
_INT8_DINOV2 = QuantScheme(
    targets=DINOV2_LINEARS,
    weights=INT8_WEIGHTS,
    input_activations=INT8_TOKEN_ACTIVATIONS,
)
_INT8_DECODER = QuantScheme(
    targets=DECODER_LINEARS,
    weights=INT8_WEIGHTS,
    input_activations=INT8_TOKEN_ACTIVATIONS,
)
_INT8_REFINERS = QuantScheme(
    targets=REFINER_POINTWISE,
    weights=INT8_WEIGHTS,
    input_activations=INT8_ACTIVATIONS,
)
_INT8_VGG = QuantScheme(
    targets=VGG_CONVS,
    weights=INT8_WEIGHTS,
    input_activations=INT8_ACTIVATIONS,
)

QUANT_CONFIGS: dict[str, QuantConfig] = {
    "fp8": QuantConfig({"transformer": _FP8_TRANSFORMER, "refiner": _FP8_REFINERS}, name="fp8"),
    "int8": QuantConfig(
        {"dinov2": _INT8_DINOV2, "decoder": _INT8_DECODER, "refiner": _INT8_REFINERS, "vgg": _INT8_VGG},
        name="int8",
    ),
}
