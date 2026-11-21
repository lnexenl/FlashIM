"""RoMa (v1) quantization schemes, protected modules, and presets.

Presets (measured on RTX 5070 Ti, sm_120, eager, default 560x560 -> 864x864, 10 MegaDepth-1500 pairs across all 5
scenes, confident pixels ``cert > 0.05`` vs ``fp32``):

- ``fp32``: checkpoint precision, no BatchNorm fold (bit-identical to upstream ``romatch`` with ``local_corr:reference``).
  Forward 483.9 ms / full ``match`` 490.9 ms.
- ``bf16``: BF16 VGG19-BN, DINOv2 ViT-L/14, coarse TransformerDecoder, and refiner conv stacks (with ``fold_batchnorm``
  and ``channels_last``); GP, projections, ``disp_emb`` and ``out_conv`` stay FP32.
  Forward 240.3 ms / ``match`` 247.9 ms; median warp error 0.022 px (p90 0.102 px, 99.49% < 2 px, cert MAE 0.0023).
- ``int8``: ``bf16`` + dynamic per-token INT8 on all 96 DINOv2 ViT-L/14 linears (``DynamicInt8Linear`` +
  ``fused_dynamic_vit_blocks`` + ``fold_layer_scale``; static INT8 fails on DINOv2 token outliers, as in ``dedode-g``)
  and static per-tensor INT8 on all 20 coarse TransformerDecoder linears (calibrated on 16 MegaDepth pairs).
  Forward 220.8 ms / ``match`` 227.9 ms; median warp error 0.023 px (p90 0.131 px, 99.48% < 2 px, cert MAE 0.0041).
- ``fp8``: ``bf16`` + static MXFP8 on all 116 transformer linears (96 DINOv2 + 20 coarse TransformerDecoder; all
  dimensions multiples of 128 -> cuBLAS block-scaled MXFP8 on Blackwell, ``fold_layer_scale`` + ``fused_block_epilogues``,
  calibrated on 16 MegaDepth pairs). Refiner 1x1 pointwise convs (non-power-of-two channels 1377 / 1137 / 569 / 144 / 24)
  stay BF16 to avoid slow im2col-GEMM autotuning.
  Forward 220.9 ms / ``match`` 228.2 ms; median warp error 0.023 px (p90 0.151 px, 99.46% < 2 px, cert MAE 0.0042).
"""

from __future__ import annotations

from pathlib import Path

from flashim.quant.scheme import (
    INT8_ACTIVATIONS,
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)

CALIBRATION_DIR = Path(__file__).parent / "calibration"

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

# Static MXFP8 on all transformer linears (DINOv2 ViT-L/14 + 5-layer coarse TransformerDecoder; all dims
# are multiples of 128 -> cuBLAS block-scaled MXFP8 on Blackwell).
_FP8_TRANSFORMER = QuantScheme(
    targets=(*DINOV2_LINEARS, *DECODER_LINEARS),
    weights=MXFP8,
    input_activations=MXFP8,
)

# DINOv2 ViT-L/14 has token-level outliers ("massive activations", same as dedode-g), so it uses dynamic
# per-token INT8 (no calibration); the coarse TransformerDecoder uses static per-tensor INT8.
_INT8_DINOV2 = QuantScheme(
    targets=DINOV2_LINEARS,
    weights=INT8_WEIGHTS,
    input_activations=INT8_TOKEN_ACTIVATIONS,
)
_INT8_DECODER = QuantScheme(
    targets=DECODER_LINEARS,
    weights=INT8_WEIGHTS,
    input_activations=INT8_ACTIVATIONS,
)

QUANT_CONFIGS: dict[str, QuantConfig] = {
    "fp8": QuantConfig({"transformer": _FP8_TRANSFORMER}, name="fp8"),
    "int8": QuantConfig({"dinov2": _INT8_DINOV2, "decoder": _INT8_DECODER}, name="int8"),
}
