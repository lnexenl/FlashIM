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


def _load_rdd(**kwargs):
    from flashim.models.rdd.build import load_rdd

    return load_rdd(**kwargs)


def _load_lightglue(**kwargs):
    from flashim.models.lightglue.build import load_lightglue

    return load_lightglue(**kwargs)


def _load_rdd_lightglue(**kwargs):
    from flashim.models.lightglue.build import load_rdd_lightglue

    return load_rdd_lightglue(**kwargs)


def _load_aliked(**kwargs):
    from flashim.models.aliked.build import load_aliked

    return load_aliked(**kwargs)


def _load_aliked_lightglue(**kwargs):
    from flashim.models.lightglue.build import load_aliked_lightglue

    return load_aliked_lightglue(**kwargs)


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

register_model(
    ModelEntry(
        name="rdd",
        kind="sparse",
        loader=_load_rdd,
        presets=("fp32", "bf16", "fp8", "fp8-full", "fp8-backbone"),
        default_preset="fp8",
        description="RDD sparse detector/descriptor (ConvNeXt-tiny + multi-scale deformable transformer)",
    )
)

register_model(
    ModelEntry(
        name="lightglue",
        kind="matcher",
        loader=_load_lightglue,
        presets=("fp32", "bf16", "fp8", "int8"),
        default_preset="fp8",
        description="LightGlue sparse matcher (RDD release weights): SparseFeatures x SparseFeatures -> matches",
    )
)

register_model(
    ModelEntry(
        name="rdd-lightglue",
        kind="sparse",
        loader=_load_rdd_lightglue,
        presets=("fp32", "bf16", "fp8", "int8"),
        default_preset="fp8",
        description="RDD detector/descriptor + LightGlue matcher (preset applies to LightGlue; rdd_preset to RDD)",
    )
)

register_model(
    ModelEntry(
        name="aliked",
        kind="sparse",
        loader=_load_aliked,
        presets=("fp32", "bf16"),
        default_preset="fp32",
        description="ALIKED sparse detector/descriptor (deformable-conv encoder + SDDH); variant=aliked-n16 default",
    )
)


def _load_raco_aliked_lightglue(**kwargs):
    from flashim.models.lightglue.build import load_raco_aliked_lightglue

    return load_raco_aliked_lightglue(**kwargs)


register_model(
    ModelEntry(
        name="raco-aliked-lightglue",
        kind="sparse",
        loader=_load_raco_aliked_lightglue,
        presets=("fp32", "bf16", "fp8", "int8"),
        default_preset="fp32",
        description="RaCo detector+ranker keypoints, ALIKED descriptors, LightGlue (ALIKED-trained) matcher",
    )
)

register_model(
    ModelEntry(
        name="aliked-lightglue",
        kind="sparse",
        loader=_load_aliked_lightglue,
        presets=("fp32", "bf16", "fp8", "int8"),
        default_preset="fp32",
        description="ALIKED detector/descriptor + LightGlue matcher (cvg/LightGlue's ALIKED-trained weights)",
    )
)
