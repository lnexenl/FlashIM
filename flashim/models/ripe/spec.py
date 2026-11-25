"""RIPE's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from flashim.config import RunConfig
from flashim.models.dedode.spec import (
    DECODER_FUSION_PASSES,
    DECODER_UPSAMPLE_PASS,
    STEM_EXACT_PASSES,
)
from flashim.models.ripe.model import RIPE
from flashim.models.sparse_spec import extractor_spec

# Same structure as DeDoDe's detector: quantize the VGG convs after the first and every ConvRefiner 1x1 (the depthwise
# 5x5s stay BF16); protected: the first conv, ``out_conv`` (heatmap logits) and the descriptor reduction (960 -> 256).
# The hidden blocks the ``fused_bf16_blocks`` pass replaces (64 / 128 channels) are gone before quantization, so their
# ``.3`` convs are not quantized; the 256 / 512-channel ones are.
QUANTIZED = ("net.encoder.layers.[1-9]*", "net.decoder.layers.*.block1.0", "net.decoder.layers.*.block1.3", "net.decoder.layers.*.hidden_blocks.*.3")
PROTECTED = ("net.encoder.layers.0", "net.decoder.layers.*.out_conv", "conv_dim_reduction_coarse_desc")


def _load(config: RunConfig) -> RIPE:
    return RIPE.from_pretrained(checkpoint=config.checkpoint, device=config.device, **config.arch)


# Keep only the bit-exact INT8 passes on RIPE (``int8_refiner_input``, ``fused_pool_epilogue``,
# ``fused_first_conv_epilogue``): the 3 non-exact DeDoDe passes (``fold_refiner_residual``, ``int8_decoder_producers``,
# ``tensor_core_first_conv``) shift RIPE's heatmap logits by 5.9% relL2 (8% keypoint flips at the 0.5 threshold) and
# dropped ScanNet-1500 int8 AUC@5 from 0.1505 (-0.0002 vs bf16, exact match to dev) to 0.1471 (-0.0036; -0.0046 with
# TF32 MNN). Exact passes alone still run 2.20x faster than dev int8 (19.19 vs 42.28 ms at 1200 px; 1.64x vs bf16).
_PASSES = {
    **DECODER_UPSAMPLE_PASS,
    "int8_refiner_input": DECODER_FUSION_PASSES["int8_refiner_input"],
    **STEM_EXACT_PASSES,
}

# Early VGG stem convs (64/128 ch) hit E4M3's 3-bit mantissa floor (~2.6% input relL2 even with per-pixel dynamic
# MXFP8 or channel smoothing), which RIPE's hypercolumn projection amplifies (ScanNet desc_cos 0.9880 -> 0.9991,
# heat_relL2 0.1358 -> 0.0434 when kept BF16; 32/35 convs remain MXFP8). INT8's 7-bit mantissa keeps them quantized.
_FP8_IGNORE = ("net.encoder.layers.3", "net.encoder.layers.7", "net.encoder.layers.10")

_SPEC = extractor_spec(
    name="ripe",
    description="RIPE detector/descriptor (VGG19 + ConvRefiner heatmap, hypercolumn 256-d descriptors, mutual-NN matching)",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("net",),
    # 16 ConvRefiner hidden blocks (64 / 128 channels) as one BF16 kernel each: BF16 extract 55.5 -> 36.8 ms
    fused_bf16_blocks=("net.decoder.",),
    channels_last=True,  # BF16 extract at 1200 px, interleaved in one process (sm_120): 76 -> 56 ms
    extra_passes=_PASSES,
    structural_passes=tuple(_PASSES),
)
SPEC = dataclasses.replace(
    _SPEC,
    quant_configs={
        **_SPEC.quant_configs,
        "fp8": dataclasses.replace(_SPEC.quant_configs["fp8"], ignore=_FP8_IGNORE),
    },
)
