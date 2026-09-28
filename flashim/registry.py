"""Model registry behind :func:`flashim.load` / :func:`flashim.build`.

Each model registers its :class:`~flashim.build.ModelSpec` (imported lazily) and, for the keyword API, its
``load_*`` function: ``flashim.load(name, preset, engine=..., ...)`` keeps working through those; the config API is
``flashim.load(name, preset, config=..., overrides=[...])``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from flashim.build import register_spec

SPECS = {
    "romav2": "flashim.models.romav2.spec:SPEC",
    "rdd": "flashim.models.rdd.spec:SPEC",
    "lightglue": "flashim.models.lightglue.spec:SPEC",
    "aliked": "flashim.models.aliked.spec:SPEC",
    "raco": "flashim.models.raco.spec:SPEC",
    "raco-aliked": "flashim.models.raco.spec:RACO_ALIKED",
    "rdd-lightglue": "flashim.models.lightglue.spec:RDD_LIGHTGLUE",
    "aliked-lightglue": "flashim.models.lightglue.spec:ALIKED_LIGHTGLUE",
    "raco-aliked-lightglue": "flashim.models.lightglue.spec:RACO_ALIKED_LIGHTGLUE",
}

# Keyword-API loaders (``module:function``).
LEGACY_LOADERS = {
    "romav2": "flashim.models.romav2.build:load_romav2",
    "rdd": "flashim.models.rdd.build:load_rdd",
    "lightglue": "flashim.models.lightglue.build:load_lightglue",
    "aliked": "flashim.models.aliked.build:load_aliked",
    "raco-aliked": "flashim.models.raco.build:load_raco_aliked",
    "rdd-lightglue": "flashim.models.lightglue.build:load_rdd_lightglue",
    "aliked-lightglue": "flashim.models.lightglue.build:load_aliked_lightglue",
    "raco-aliked-lightglue": "flashim.models.lightglue.build:load_raco_aliked_lightglue",
}

for _name, _target in SPECS.items():
    register_spec(_name, _target)


def legacy_loader(name: str) -> Callable:
    if name not in LEGACY_LOADERS:
        raise ValueError(f"Model {name!r} has no keyword API; use config= / overrides=")
    module, attr = LEGACY_LOADERS[name].split(":")
    return getattr(importlib.import_module(module), attr)
