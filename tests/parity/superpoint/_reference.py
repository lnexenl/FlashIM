"""cvg/LightGlue's ``lightglue.SuperPoint`` as the reference of the SuperPoint parity test, fetched on demand.

``lightglue/superpoint.py`` carries a Magic Leap notice that forbids reproduction and distribution without the
company's written permission (the rest of LightGlue is Apache-2.0), so it is not copied into this repository
(see ``tests/parity/_fetch.py``). The package ``__init__.py`` is ours: upstream's imports every extractor.
"""

from __future__ import annotations

from pathlib import Path

from .._fetch import fetch

COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
FILES = {
    "lightglue/superpoint.py": "23e8a26137b44f37e9a319313822c5b38d9c55796e756202eb46d64f1979d2b0",
    "lightglue/utils.py": "c6d9260656a24d826128b68c95d0b7798106d04f4244a77f3b1856573f58e690",
}


def reference_root() -> Path:
    """The directory that holds the ``lightglue`` package (``superpoint`` and ``utils`` only)."""
    root = fetch("cvg/LightGlue", COMMIT, FILES, "lightglue-superpoint")
    (root / "lightglue" / "__init__.py").touch()
    return root
