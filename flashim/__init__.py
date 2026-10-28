"""FlashIM: fast, quantized, multi-platform inference for image matching.

    import flashim
    model = flashim.load("romav2", "fp8", engine="compile")      # preset: fp32 / bf16 / int8 / fp8
    model = flashim.load("rdd-lightglue", "int8", overrides=["inference.resize=1024"])
    result = model.match("a.jpg", "b.jpg", max_matches=5000)      # Matches, for every model
    result.keypoints_A, result.keypoints_B, result.confidence

``flashim.resolve(name, preset, config=, overrides=)`` returns the :class:`~flashim.config.RunConfig` a build would
use; a built model keeps it as ``model.run_config`` and reports what was applied in ``model.build_report``.
"""

from flashim.build import PRECISIONS, build_config, get_spec, load, resolve, spec_names
from flashim.config import RunConfig
from flashim.log import set_verbosity
from flashim.registry import SPECS as _SPECS  # noqa: F401  (registers the models)
from flashim.types import DenseMatchResult, Matches, SparseFeatures

__version__ = "0.0.1"


def available_models() -> dict[str, str]:
    return {name: get_spec(name).description for name in spec_names()}


__all__ = [
    "PRECISIONS",
    "DenseMatchResult",
    "Matches",
    "RunConfig",
    "SparseFeatures",
    "available_models",
    "build_config",
    "load",
    "resolve",
    "set_verbosity",
]
