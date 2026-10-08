"""Reference code that cannot be copied into this repository, fetched on demand for the parity tests.

The upstream licenses of these files grant internal use only (RIPE: Fraunhofer academic-use license) or carry a
no-redistribution notice (SuperPoint: Magic Leap header in cvg/LightGlue), so the files are downloaded from GitHub into
``~/.cache/flashim/oracles`` when a test needs them, for your own evaluation and testing. Each file is pinned by commit
and SHA-256; a hash mismatch or a failed download skips the test.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import pytest


def fetch(repo: str, commit: str, files: dict[str, str], name: str) -> Path:
    """Download ``files`` (path -> SHA-256) of github.com/``repo`` at ``commit``; return the directory holding them."""
    cache = Path(os.environ.get("FLASHIM_CACHE", Path.home() / ".cache" / "flashim"))
    root = cache / "oracles" / f"{name}-{commit[:10]}"
    for rel, digest in files.items():
        target = root / rel
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        try:
            data = urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}", timeout=60).read()
        except OSError as exc:
            pytest.skip(f"cannot download {repo}:{rel} ({exc})")
        if hashlib.sha256(data).hexdigest() != digest:
            pytest.skip(f"downloaded {repo}:{rel} does not match the pinned SHA-256")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root
