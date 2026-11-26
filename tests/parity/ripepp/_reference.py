"""Upstream RIPE++ (fraunhoferhhi/RIPEpp) as the reference of the RIPE++ parity test, never vendored here.

RIPE++'s license (Fraunhofer "Software Copyright License for Academic Use of RIPE++, Version 2.0") grants internal
non-commercial use for evaluation, testing and academic research only, with no redistribution rights, so its source
code is not copied into ``tests/oracles/``. Lookup order:

1. ``$FLASHIM_RIPEPP_ROOT`` pointing at a local checkout;
2. an already-installed ``ripepp`` package (``importlib.util.find_spec``);
3. on-demand fetch into ``~/.cache/flashim/oracles`` pinned by commit and SHA-256 (:mod:`tests.parity._fetch`),
   skipping cleanly if offline.

Upstream training-only imports (``hydra``, ``omegaconf``, ``lightning``) are stubbed narrowly during module import
when absent from the test environment so the unmodified inference classes (``VGG``, ``KeypointSampler``, ``RIPEPP``)
run without installing extra dependencies.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from .._fetch import fetch

COMMIT = "0666e62bf00569870bbcc1fe4b2e03976fc563f8"
FILES = {
    "ripepp/__init__.py": "858610bf0dc57bf06af20a6a79e5a83430536d255860ea03e3ff38c87f38ff47",
    "ripepp/models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "ripepp/models/ripepp.py": "4a4139f36dfbf39ced1bbf9b1c868f5956a7a85228d497368757a944b413504b",
    "ripepp/models/backbones/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "ripepp/models/backbones/backbone_base.py": "dd91fd6b102a6790d1e2ea5e0b8db19eeceea7c0dd37b7e4e4f6ea35dc74a580",
    "ripepp/models/backbones/vgg.py": "0c1f165ee9fc719e63982d16b9c561102efcc4cecdbfebd36ae649fbda537e9a",
    "ripepp/models/backbones/vgg_utils.py": "b44e06eb01f23dedbd141eda5673ad294205be169c3dfe9df7b310feecdd999c",
    "ripepp/models/sampler/base_sampler.py": "31cc8054851b600b8a96dabdf0301374100d99596fcdadb9efe763b2024ab205",
    "ripepp/models/sampler/keypoint_sampler.py": "e6284789092b98069b51bb3efbdf2a990bf671f9839017219dad37515bd6c9f5",
    "ripepp/models/upsampler/direct_sample.py": "cd9971eff0f4947214043b380a72dff3099a97552b7fc7beddd7700202943104",
    "ripepp/models/upsampler/hypercolumn_features.py": "7c853d8d95a75ef01ed600f46ff1f92c73590e92fbab1e61bff02b159c65a7e6",
    "ripepp/utils/__init__.py": "7776e3356759297f464e4640b9598ec0a25900f4c0ec3471cd2ff7feb302a62d",
    "ripepp/utils/checkpoint.py": "f7251002f8bf551a8aabba6215c8e3dd5b5e457a5edbb76feb7186224ae2ba4e",
    "ripepp/utils/pylogger.py": "2b52479406eb1bf7aab9b019ec4ba2c96fc1986c32ac43e280d9a7a917aad2ad",
    "ripepp/utils/utils.py": "b702df57c5171b50c24e484c4c9c34efe94ccfd938c515fbd56f7bbc7c14c4f2",
}


def reference_root() -> Path | None:
    """Directory holding ``ripepp/``, or ``None`` if ``ripepp`` is already installed on ``sys.path``."""
    if os.environ.get("FLASHIM_RIPEPP_ROOT"):
        root = Path(os.environ["FLASHIM_RIPEPP_ROOT"]).expanduser()
        if not (root / "ripepp").is_dir():
            pytest.skip(f"FLASHIM_RIPEPP_ROOT={root} has no ripepp/ package")
        return root
    if importlib.util.find_spec("ripepp") is not None:
        return None
    return fetch("fraunhoferhhi/RIPEpp", COMMIT, FILES, "ripepp")


@contextlib.contextmanager
def _stub_training_deps():
    """Temporarily stub optional training-only deps (hydra, omegaconf, lightning) if missing."""
    injected: list[str] = []

    def ensure(name: str, attrs: dict | None = None) -> None:
        if name in sys.modules:
            return
        top = name.split(".")[0]
        if top not in injected and importlib.util.find_spec(top) is not None:
            return
        mod = types.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        injected.append(name)

    ensure("hydra")
    ensure("hydra.utils", {"instantiate": None})
    ensure("omegaconf", {"DictConfig": object, "OmegaConf": object})
    ensure("lightning")
    ensure("lightning.fabric")
    ensure("lightning.fabric.utilities", {"rank_zero_only": lambda fn: fn})
    try:
        yield
    finally:
        for name in injected:
            sys.modules.pop(name, None)


def import_upstream_classes():
    """Return ``(VGG, KeypointSampler, RIPEPP)`` from the unmodified upstream ``ripepp`` package."""
    root = reference_root()
    if root is not None:
        sys.path.insert(0, str(root))
    try:
        with _stub_training_deps():
            from ripepp.models.backbones.vgg import VGG
            from ripepp.models.ripepp import RIPEPP
            from ripepp.models.sampler.keypoint_sampler import KeypointSampler
    finally:
        if root is not None and str(root) in sys.path:
            sys.path.remove(str(root))
    return VGG, KeypointSampler, RIPEPP
