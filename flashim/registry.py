"""Model registry behind :func:`flashim.load`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    name: str
    kind: str  # "dense" | "sparse"
    loader: Callable[..., object]
    presets: tuple[str, ...]
    default_preset: str
    description: str


MODELS: dict[str, ModelEntry] = {}


def register_model(entry: ModelEntry) -> None:
    if entry.name in MODELS:
        raise ValueError(f"Duplicate model {entry.name!r}")
    MODELS[entry.name] = entry


def _load_romav2(**kwargs):
    from flashim.models.romav2.build import load_romav2

    return load_romav2(**kwargs)


register_model(
    ModelEntry(
        name="romav2",
        kind="dense",
        loader=_load_romav2,
        presets=("bf16", "fp8-p8", "fp8-s4", "fp8", "fp8-fast"),
        default_preset="fp8",
        description="RoMa v2 dense matcher (DINOv3 ViT-L/16 + MV-ViT + DPT + conv refiners)",
    )
)
