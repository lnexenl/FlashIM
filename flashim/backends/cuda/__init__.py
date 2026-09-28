"""CUDA C++ kernels, JIT-compiled with ``nvcc`` on first use.

Sources live in ``csrc/``; each is built into a shared library cached under ``~/.cache/flashim/cuda`` (keyed by a
hash of the source, the target architecture and the compiler), and launched through ``ctypes`` on PyTorch's current
stream — no torch C++ headers, so no dependence on the host compiler's compatibility with them. Every CUDA kernel has a
Triton/torch fallback: :func:`load` returns ``None`` (with one logged warning) when there is no toolchain or the build
fails, and callers must then use it.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import os
import subprocess
from pathlib import Path

import torch

from flashim.platforms.nvidia import nvcc_path

logger = logging.getLogger(__name__)
_SOURCE_DIR = Path(__file__).parent / "csrc"
_FLAGS = ("-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-allow-unsupported-compiler")


def _cache_dir() -> Path:
    return Path(os.environ.get("FLASHIM_CACHE", Path.home() / ".cache" / "flashim")) / "cuda"


@functools.cache
def load(name: str, capability: tuple[int, int] | None = None) -> ctypes.CDLL | None:
    """Build (or fetch from the cache) ``<name>.cu`` for ``capability`` (default: the current device) and load it;
    ``None`` if it cannot be built."""
    source = _SOURCE_DIR / f"{name}.cu"
    nvcc = nvcc_path()
    if nvcc is None:
        logger.warning("nvcc not found: CUDA C++ kernel %r unavailable, using the fallback", name)
        return None
    major, minor = capability or torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    key = hashlib.sha256(source.read_bytes() + f"{arch}{nvcc}{_FLAGS}".encode()).hexdigest()[:16]
    library = _cache_dir() / f"{name}-{arch}-{key}.so"
    if not library.is_file():
        library.parent.mkdir(parents=True, exist_ok=True)
        temporary = library.with_suffix(f".{os.getpid()}.tmp")
        result = subprocess.run(
            [nvcc, *_FLAGS, f"-arch={arch}", "-o", str(temporary), str(source)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            logger.warning("building CUDA kernel %r failed, using the fallback:\n%s", name, result.stderr[-2000:])
            return None
        temporary.replace(library)  # atomic: concurrent processes build the same content
    return ctypes.CDLL(str(library))
