"""ORB's :class:`~flashim.build.ModelSpec`: ``fp32`` only (OpenCV, no weights: nothing to run in bf16 / int8 / fp8)."""

from __future__ import annotations

from flashim.build import EAGER, ModelSpec
from flashim.config import RunConfig
from flashim.models.orb.model import ORB


def _load(config: RunConfig) -> ORB:
    return ORB.from_pretrained(device=config.device, **config.arch)


SPEC = ModelSpec(
    name="orb",
    role="extractor",
    description="ORB (OpenCV, CPU detection/description; GPU Hamming ratio-test mutual-NN matching); fp32 only",
    load=_load,
    presets={"fp32": {}},
    default_preset="fp32",
    engines={"eager": EAGER},
)
