"""FlashIM: fast, quantized, multi-platform inference for image matching.

    import flashim
    model = flashim.load("romav2", preset="fp8", setting="mega1500", engine="compile")      # keyword API
    model = flashim.load("romav2", "fp8-fast", overrides=["engine.kind=compile-full"])    # config API
    result = model.match("a.jpg", "b.jpg")          # DenseMatchResult
    matches, conf, prec_a, prec_b = model.sample(result, 5000)

``flashim.resolve(name, preset, config=, overrides=)`` returns the :class:`~flashim.config.RunConfig` a build would
use; a built model keeps it as ``model.run_config`` and reports what was applied in ``model.build_report``.
"""

from flashim.build import build, build_config, get_spec, resolve, spec_names
from flashim.config import RunConfig
from flashim.log import set_verbosity
from flashim.registry import legacy_loader
from flashim.types import DenseMatchResult

__version__ = "0.0.1"


def load(name: str, preset: str | None = None, *, config=None, overrides=(), **kwargs):
    """Load a registered model.

    With keyword arguments (``setting=``, ``engine=``, ``device=``, ...) this calls the model's ``load_*`` function
    (the keyword API); otherwise it resolves ``preset`` + ``config`` (a :class:`RunConfig`, dict or .toml/.json
    file) + dotted ``overrides`` and builds (:func:`flashim.build`).
    """
    if kwargs:
        if config is not None or overrides:
            raise ValueError("Pass either keyword arguments or config= / overrides=, not both")
        return legacy_loader(name)(preset=preset or get_spec(name).default_preset, **kwargs)
    return build(name, preset, config=config, overrides=overrides)


def available_models() -> dict[str, str]:
    return {name: get_spec(name).description for name in spec_names()}


__all__ = [
    "DenseMatchResult",
    "RunConfig",
    "available_models",
    "build",
    "build_config",
    "load",
    "resolve",
    "set_verbosity",
]
