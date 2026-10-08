"""Upstream RIPE (fraunhoferhhi/RIPE) as the reference of the RIPE parity test, fetched on demand.

RIPE's license (Fraunhofer "Software Copyright License for Academic Use", version 2.0) grants internal non-commercial
use for evaluation, testing and academic research and nothing else, so its code is not copied into this repository
(see ``tests/parity/_fetch.py``). ``FLASHIM_RIPE_ROOT`` points at an existing checkout instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .._fetch import fetch

COMMIT = "b173418008f4cb77a2ebffb570a7cab69e87cd08"
FILES = {
    "ripe/__init__.py": "b40e131a9654b129561ee8e10fcad3df8a6f66a22b25795d060e1cb89e37a1ad",
    "ripe/model_zoo/__init__.py": "451cbd58b3ee47f209e0798d0857842bc040ea9c209c5dd899b862409ccce234",
    "ripe/model_zoo/vgg_hyper.py": "fc291cee0bd044e0ee8d01b003b55e71e5813aff010dfd6ecd2109b4d55bf83c",
    "ripe/models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "ripe/models/backbones/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "ripe/models/backbones/backbone_base.py": "dd91fd6b102a6790d1e2ea5e0b8db19eeceea7c0dd37b7e4e4f6ea35dc74a580",
    "ripe/models/backbones/vgg.py": "9ddc76742d0681e9b4c1882a2b034313bef738ce9129e304a77f12964a5e9edd",
    "ripe/models/backbones/vgg_utils.py": "2d992e82f5b6d91e02ea93e1d259350766639bef42b4ca38912b4bc7af3deda2",
    "ripe/models/ripe.py": "d382021ddceb5f8fab1059a75c2cc30240ec21d6619fbfc89b0b973e7724c50a",
    "ripe/models/upsampler/hypercolumn_features.py": "d2cec9605fad9809afd7af20d13d3582df29bd4fa2bc7a2f8814b12c2eafeb9f",
    "ripe/utils/__init__.py": "996c7189ac0f82ba7b04085e295159abd43c50edddd3dacc523a68d9a6fb13df",
    "ripe/utils/pylogger.py": "478b2522aa2a2c25637545b96f80ca1419743936c871659e3aa091b847b86774",
    "ripe/utils/utils.py": "c6fb63c71a7a4d51945ed118b14c0843fb652fedc2398e23e784d2ccc99c0d03",
}


def reference_root() -> Path:
    """The directory that holds the ``ripe`` package; skips the test when it cannot be obtained."""
    if os.environ.get("FLASHIM_RIPE_ROOT"):
        root = Path(os.environ["FLASHIM_RIPE_ROOT"]).expanduser()
        if not (root / "ripe").is_dir():
            pytest.skip(f"FLASHIM_RIPE_ROOT={root} has no ripe/ package")
        return root
    return fetch("fraunhoferhhi/RIPE", COMMIT, FILES, "ripe")
