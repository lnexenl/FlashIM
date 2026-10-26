"""Checkpoint lookup shared by the model ports: an explicit path, the model's environment variable, the torch
hub checkpoint folder (``<torch.hub.get_dir()>/checkpoints``, i.e. ``$TORCH_HOME/hub/checkpoints``), else a
download of the pinned upstream release into that folder, verified against its SHA-256."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger("flashim.weights")


@dataclass(frozen=True)
class Checkpoint:
    """An upstream release file, stored as ``filename`` in the torch hub checkpoint folder."""

    filename: str
    url: str
    sha256: str


def checkpoint_dir() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints"


def find_weights(checkpoint: Checkpoint, *, path: str | Path | None = None, env: str | None = None) -> Path:
    """``path`` or ``$env`` when set (they must exist), else the hub folder's copy, downloaded on first use.

    Only downloads are hashed: files already in place are not re-read (some are over 1 GB)."""
    for explicit, origin in ((path, "checkpoint="), (os.environ.get(env) if env else None, f"${env}")):
        if explicit:
            if not Path(explicit).is_file():
                raise FileNotFoundError(f"{origin} {str(explicit)!r} is not a file")
            return Path(explicit)
    target = checkpoint_dir() / checkpoint.filename
    if target.is_file():
        return target
    logger.warning("downloading %s -> %s", checkpoint.url, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:  # downloads to a temporary file next to the target, renamed once the hash matches
        torch.hub.download_url_to_file(checkpoint.url, str(target), hash_prefix=checkpoint.sha256)
    except Exception as error:
        hint = f", pass checkpoint= or set ${env}" if env else ""
        raise FileNotFoundError(
            f"checkpoint {checkpoint.filename!r} is missing and downloading {checkpoint.url} failed "
            f"({error}); place it at {target}{hint}"
        ) from error
    return target
