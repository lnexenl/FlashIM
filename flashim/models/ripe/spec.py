"""RIPE's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from pathlib import Path

from flashim.config import RunConfig
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


SPEC = extractor_spec(
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
)
