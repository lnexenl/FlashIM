"""DaD + DeDoDe descriptors' :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``).

DaD is a keypoint *detector* (VGG11 + ConvRefiner decoder; NMS 3, sub-pixel refinement); as a matcher it needs a
descriptor: DeDoDe-G (VGG19 + DINOv2 ViT-L/14, the pairing of the DaD paper), matched by DeDoDe's dual softmax.
"""

from __future__ import annotations

from pathlib import Path

from flashim.config import RunConfig
from flashim.models.dad import weights as dad_weights
from flashim.models.dedode import blocks
from flashim.models.dedode import weights as dedode_weights
from flashim.models.dedode.model import DAD_SAMPLER, DeDoDe
from flashim.models.dedode.spec import (
    DECODER_UPSAMPLE_PASS,
    PROTECTED,
    QUANTIZED,
    VIT_EPILOGUE_PASSES,
    VIT_IGNORE,
)
from flashim.models.sparse_spec import extractor_spec


def _load(config: RunConfig) -> DeDoDe:
    return DeDoDe.from_pretrained(
        blocks.DETECTOR_S, blocks.DESCRIPTOR_G, DAD_SAMPLER,
        detector_state=dad_weights.load_state_dict(config.checkpoint),
        descriptor_state=dedode_weights.load("descriptor-G"),
        dinov2_state=dedode_weights.load("dinov2"),
        device=config.device,
    )


SPEC = extractor_spec(
    name="dad",
    description="DaD keypoint detector (VGG11) + DeDoDe-G descriptors, dual-softmax matching",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    int8_ignore=VIT_IGNORE,  # see dedode.spec.VIT_IGNORE
    compile_regions=("detector", "descriptor"),
    # 16 ConvRefiner hidden blocks (32 / 64 channels) as one BF16 kernel each: BF16 extract 141.0 -> 133.0 ms
    fused_bf16_blocks=("detector.decoder.", "descriptor.decoder."),
    channels_last=True,  # BF16 extract at 1200 px, interleaved in one process (sm_120): 135 -> 127 ms
    extra_passes={**DECODER_UPSAMPLE_PASS, **VIT_EPILOGUE_PASSES},
    structural_passes=(*DECODER_UPSAMPLE_PASS, *VIT_EPILOGUE_PASSES),
)
