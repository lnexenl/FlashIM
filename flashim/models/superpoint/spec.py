"""SuperPoint's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``;
variants ``v1`` (default, Magic Leap) and ``sp_v6`` (rpautrat/SuperPoint with BatchNorm, used by OmniGlue))."""

from __future__ import annotations

from pathlib import Path

from flashim.config import RunConfig
from flashim.models.sparse_spec import extractor_spec
from flashim.models.superpoint.model import SuperPoint

# Quantized: every 3x3 trunk conv after the 1-channel stem and both 256-wide head convs. Protected (never
# quantized): ``conv1a`` (K = 9, image-facing) and the 65-way score head ``convPb`` (softmax logits) and
# ``convDb`` (final descriptor projection, normalized right after).
QUANTIZED = ("conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b", "convPa", "convDa")
PROTECTED = ("conv1a", "convPb", "convDb")
VARIANTS = ("v1", "sp_v6")


def _load(config: RunConfig) -> SuperPoint:
    variant = config.variant or "v1"
    return SuperPoint.from_pretrained(checkpoint=config.checkpoint, device=config.device, variant=variant, **config.arch)


def _validate(config: RunConfig) -> None:
    variant = config.variant or "v1"
    if variant not in VARIANTS:
        raise ValueError(f"superpoint: unknown variant {variant!r}; choose from {VARIANTS}")


_SPEC = extractor_spec(
    name="superpoint",
    description="SuperPoint detector/descriptor (VGG encoder, 256-d descriptors; variant v1 | sp_v6)",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("conv1a", "conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b"),
    has_batchnorm=True,  # no-op for v1 (empty batchnorm_pairs), folds convPb/convDb BN for sp_v6
    channels_last=True,  # BF16 extract at 1200 px, interleaved in one process (sm_120): 7.8 -> 6.7 ms
)
import dataclasses

SPEC = dataclasses.replace(_SPEC, validate=_validate)
