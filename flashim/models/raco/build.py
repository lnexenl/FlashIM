"""Build the RaCo + ALIKED extractor (``load_raco_aliked``: the keyword API over :mod:`flashim.build`)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from flashim.build import build
from flashim.models.aliked.spec import legacy_overrides as aliked_overrides
from flashim.models.raco.config import RaCoConfig
from flashim.models.raco.extractor import RaCoALIKED

PRESETS = ("fp32", "bf16")


def raco_aliked_overlay(
    preset: str = "fp32",
    *,
    engine: str = "eager",
    raco_checkpoint: str | Path | None = None,
    raco_config: RaCoConfig | None = None,
    aliked_kwargs: dict | None = None,
    fused_ranker: bool = True,
) -> dict:
    """``load_raco_aliked`` keyword arguments -> a ``raco-aliked`` config overlay."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {PRESETS}")
    raco: dict = {"model": "raco", "preset": preset}
    if not fused_ranker:
        raco["optimize"] = {"passes": ["fold_batchnorm"]}
    if raco_checkpoint is not None:
        raco["checkpoint"] = str(raco_checkpoint)
    if raco_config is not None:
        raco["arch"] = dataclasses.asdict(raco_config)
    aliked = {"model": "aliked", "preset": preset, **aliked_overrides(engine=engine, **(aliked_kwargs or {}))}
    return {"engine": {"kind": engine}, "components": {"raco": raco, "aliked": aliked}}


def load_raco_aliked(
    preset: str = "fp32",
    *,
    engine: str = "eager",
    device: torch.device | str | None = None,
    raco_checkpoint: str | Path | None = None,
    raco_config: RaCoConfig | None = None,
    aliked_kwargs: dict | None = None,
    fused_ranker: bool = True,
) -> RaCoALIKED:
    """RaCo keypoints + ALIKED (``aliked-n16``) descriptors. ``preset`` sets both precisions; RaCo's detector trunk
    and ranker follow it (the ranker as fused ops when ``fused_ranker``, CUDA only) while its score head and
    NMS/soft-argmax stay FP32. Equivalent to ``flashim.build("raco-aliked", preset, config=...)``."""
    overlay = raco_aliked_overlay(
        preset, engine=engine, raco_checkpoint=raco_checkpoint, raco_config=raco_config,
        aliked_kwargs=aliked_kwargs, fused_ranker=fused_ranker,
    )
    if device is not None:
        overlay["device"] = str(device)
    return build("raco-aliked", preset, config=overlay)
