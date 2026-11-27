"""XFeat's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from pathlib import Path

from flashim.config import RunConfig
from flashim.models.sparse_spec import QUANT_COMPILE_FAILURES, extractor_spec
from flashim.models.xfeat.model import XFeat

# Quantized: the 64/128-channel trunk convolutions (blocks 3-5), the fusion block and the reliability heatmap trunk.
# Protected: the 1/4-resolution stem (``block1``, ``skip1``, ``block2``: 1-24 channels, not block-32-aligned; keeping
# ``block2`` BF16 cuts FP8 descriptor cosine error 3x from 0.9942 -> 0.9980 and INT8 0.9985 -> 0.9995), the
# standalone image-facing keypoint detector ``keypoint_head.*`` (three 64->64 1x1 convs on unfolded raw pixels feeding
# the 65-way cell softmax; quantizing it dropped MegaDepth keypoint overlap to 0.517 FP8 / 0.729 INT8 and caused a
# -0.0086 dev / -0.0137 integration FP8 MegaDepth AUC@5 regression; protecting it restores 0.969 FP8 / 0.987 INT8
# overlap while keeping extract faster than BF16), the fusion output projection (feeds the L2-normalized descriptor)
# and the final output convs.
QUANTIZED = ("block[345].*.layer.0", "block_fusion.[01].layer.0", "heatmap_head.[01].layer.0")
PROTECTED = (
    "block1.*", "skip1.*", "block2.*", "block_fusion.2", "heatmap_head.2", "keypoint_head.*", "fine_matcher.*",
)


def _load(config: RunConfig) -> XFeat:
    return XFeat.from_pretrained(checkpoint=config.checkpoint, device=config.device, **config.arch)


SPEC = extractor_spec(
    name="xfeat",
    description="XFeat detector/descriptor (0.7M-parameter conv net, 64-d descriptors, mutual-NN matching at 0.82)",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("skip1", "block1", "block2", "block3", "block4", "block5", "block_fusion", "heatmap_head", "keypoint_head"),
    compile_failures=QUANT_COMPILE_FAILURES,
    cuda_graph=True,
)
