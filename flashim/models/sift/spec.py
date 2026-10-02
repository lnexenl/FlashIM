"""SIFT's :class:`~flashim.build.ModelSpec`: ``fp32`` only (OpenCV, no weights: nothing to run in bf16 / int8 / fp8)."""

from __future__ import annotations

from flashim.build import EAGER, ModelSpec
from flashim.config import RunConfig
from flashim.models.sift.model import SIFT


def _load(config: RunConfig) -> SIFT:
    return SIFT.from_pretrained(device=config.device, **config.arch)


SPEC = ModelSpec(
    name="sift",
    role="extractor",
    description="SIFT (OpenCV, CPU detection/description; GPU ratio-test mutual-NN matching); fp32 only",
    load=_load,
    presets={"fp32": {}},
    default_preset="fp32",
    engines={"eager": EAGER},
)
