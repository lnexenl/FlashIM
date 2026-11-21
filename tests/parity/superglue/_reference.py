"""Upstream SuperGlue (magicleap/SuperGluePretrainedNetwork) fetched on demand for the parity test.

``models/superglue.py`` carries a Magic Leap notice that forbids reproduction and distribution without the
company's written permission (same notice as SuperPoint), so it is not copied into this repository
(see ``tests/parity/_fetch.py``). ``FLASHIM_SUPERGLUE_ROOT`` points at an existing checkout instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from flashim.models.superglue import weights

from .._fetch import fetch

COMMIT = "ddcf11f42e7e0732a0c4607648f9448ea8d73590"
FILES = {
    "models/superglue.py": "14b2d1bc363fef84c79653df5660389eeb505250cad8faf7507d604ced0a9fad",
}


def reference_root(variant: str = "outdoor") -> Path:
    """Directory holding ``models/superglue.py`` with ``models/weights/superglue_<variant>.pth`` linked in."""
    if os.environ.get("FLASHIM_SUPERGLUE_ROOT"):
        root = Path(os.environ["FLASHIM_SUPERGLUE_ROOT"]).expanduser()
        if not (root / "models" / "superglue.py").is_file():
            pytest.skip(f"FLASHIM_SUPERGLUE_ROOT={root} has no models/superglue.py")
    else:
        root = fetch("magicleap/SuperGluePretrainedNetwork", COMMIT, FILES, "superglue")
    (root / "models" / "__init__.py").touch()
    checkpoint = weights.find_checkpoint(variant)
    weights_dir = root / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / f"superglue_{variant}.pth"
    if not target.exists():
        target.symlink_to(checkpoint)
    return root
