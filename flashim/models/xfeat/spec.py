"""XFeat's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from pathlib import Path

from flashim.config import RunConfig
from flashim.models.sparse_spec import extractor_spec
from flashim.models.xfeat.model import XFeat

# Quantized: the convolutions of blocks 2-5, the fusion block and the head trunks (24-128 channels). Protected:
# the 1/8-resolution stem (``block1``, ``skip1``: 1-8 input channels, image-facing), the fusion output projection
# (feeds the L2-normalized descriptor) and the two output convs (keypoint logits / reliability, FP32).
QUANTIZED = (
    "block[2345].*.layer.0", "block_fusion.[01].layer.0", "heatmap_head.[01].layer.0", "keypoint_head.[012].layer.0",
)
PROTECTED = ("block1.*", "skip1.*", "block_fusion.2", "heatmap_head.2", "keypoint_head.3", "fine_matcher.*")


def _load(config: RunConfig) -> XFeat:
    return XFeat.from_pretrained(checkpoint=config.checkpoint, device=config.device, **config.arch)


SPEC = extractor_spec(
    name="xfeat",
    description="XFeat detector/descriptor (0.7M-parameter conv net, 64-d descriptors, mutual-NN matching at 0.82)",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("block1", "block2", "block3", "block4", "block5", "block_fusion"),
)
