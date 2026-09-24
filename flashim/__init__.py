"""FlashIM: fast, quantized, multi-platform inference for image matching.

    import flashim
    model = flashim.load("romav2", preset="fp8", setting="mega1500", engine="compile")
    result = model.match("a.jpg", "b.jpg")          # DenseMatchResult
    matches, conf, prec_a, prec_b = model.sample(result, 5000)
"""

from flashim.registry import MODELS
from flashim.types import DenseMatchResult

__version__ = "0.0.1"


def load(name: str, preset: str | None = None, **kwargs):
    """Load a registered model. ``kwargs`` go to the model's loader (setting, engine, device, ...)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODELS)}")
    entry = MODELS[name]
    return entry.loader(preset=preset or entry.default_preset, **kwargs)


def available_models() -> dict[str, str]:
    return {name: entry.description for name, entry in MODELS.items()}


__all__ = ["DenseMatchResult", "available_models", "load"]
