"""ALIKE's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from flashim.config import RunConfig
from flashim.models.alike.config import VARIANTS
from flashim.models.alike.model import ALIKE
from flashim.models.sparse_spec import extractor_spec

DEFAULT_VARIANT = "alike-n"

# Quantized: every trunk conv after the 3-channel stem, the aggregation 1x1s and ``convhead1``. Protected: ``block1.conv1``
# (K = 27, image-facing) and the descriptor/score head ``convhead2``.
QUANTIZED = (
    "block1.conv2", "block[234].conv[12]", "block[234].downsample", "conv[1234]", "convhead1",
)
PROTECTED = ("block1.conv1", "convhead2")


def _load(config: RunConfig) -> ALIKE:
    variant = config.variant or DEFAULT_VARIANT
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    arch = dataclasses.replace(VARIANTS[variant], **config.arch) if config.arch else None
    return ALIKE.from_pretrained(arch, variant=variant, checkpoint=config.checkpoint, device=config.device)


SPEC = extractor_spec(
    name="alike",
    description="ALIKE detector/descriptor (variant alike-t | alike-s | alike-n | alike-l)",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("block1", "block2", "block3", "block4"),
)
