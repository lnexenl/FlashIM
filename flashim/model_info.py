"""License metadata for registered models and their default checkpoints."""

from __future__ import annotations

from importlib import import_module

from flashim.build import resolve
from flashim.config import RunConfig
from flashim.models.weights_util import ModelInfo
from flashim.registry import SPECS

__all__ = ["ModelInfo", "model_info"]


def _component_info(config: RunConfig) -> ModelInfo:
    if config.components:
        parts = [_component_info(part) for part in config.components.values()]
        return ModelInfo(
            tuple(sorted({term for part in parts for term in part.code_licenses})),
            tuple(sorted({term for part in parts for term in part.weights_licenses})),
        )
    if config.model in {"orb", "sift"}:
        return ModelInfo(("Apache-2.0",), ())
    if config.model not in SPECS:
        return ModelInfo(("NOASSERTION",), ("NOASSERTION",))
    package = SPECS[config.model].split(":")[0].rpartition(".")[0]
    weights = import_module(f"{package}.weights")
    variants = getattr(weights, "MODEL_INFO_OVERRIDES", {})
    return variants.get(config.variant or config.model, weights.MODEL_INFO)


def model_info(name: str) -> ModelInfo:
    """Return the model and upstream code and weight license IDs for the default configuration, offline.

    Pipelines include all components. NOASSERTION means no license was recorded for that artifact.
    An empty weights tuple means the model needs no weights.
    """
    return _component_info(resolve(name))
