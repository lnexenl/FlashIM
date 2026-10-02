"""Checkpoint lookup shared by the model ports: explicit path, environment variable, torch hub cache, then the
sibling ``~/work/oracles/weights`` folder the upstream checkpoints were downloaded to."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch

ORACLE_WEIGHTS = Path.home() / "work" / "oracles" / "weights"


def find_weights(filename: str, *, path: str | Path | None = None, env: str | None = None, help: str = "",
                 extra: Iterable[str | Path] = ()) -> Path:
    candidates = [
        path,
        os.environ.get(env) if env else None,
        Path(torch.hub.get_dir()) / "checkpoints" / filename,
        ORACLE_WEIGHTS / filename,
        *extra,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(f"checkpoint {filename!r} not found (set {env or 'checkpoint='}). {help}")
