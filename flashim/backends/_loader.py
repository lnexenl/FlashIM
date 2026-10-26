"""Which backend modules exist and loading them (called once, at the end of ``import flashim.ops``)."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from importlib.metadata import entry_points

logger = logging.getLogger("flashim.backends")


@dataclass(frozen=True)
class Backend:
    """``modules`` (relative to :mod:`flashim.backends`) register impls when imported; they are imported only if
    every Python package in ``packages`` is installed. Hardware requirements are per impl (``requires=``)."""

    name: str
    modules: tuple[str, ...]
    packages: tuple[str, ...] = ()

    def installed(self) -> bool:
        return all(importlib.util.find_spec(package) is not None for package in self.packages)


BACKENDS: tuple[Backend, ...] = (
    Backend("torch_native", ("torch_native.gemm",)),
    Backend(
        "triton",
        tuple(f"triton.{m}" for m in ("aliked", "attention", "conv", "correlation", "density", "gemm", "quantize", "raco", "resample", "romav2")),
        packages=("triton",),
    ),
    Backend("tilelang", ("tilelang.conv", "tilelang.gemm"), packages=("tilelang",)),
    Backend("cuda", ("cuda.raco",)),  # JIT CUDA C++: needs nvcc (Feature.NVCC), built on first eligible call
    Backend("sageattention", ("external.sageattention",), packages=("sageattention",)),
)

LOADED: list[str] = []
_done = False


def load() -> None:
    """Import every installed in-tree backend, then the ``flashim.backends`` entry points (out-of-tree backends:
    the entry point's object is a module or callable that registers impls). A failing backend is logged and
    skipped: FlashIM keeps working with the others and the torch references."""
    global _done
    if _done:
        return
    _done = True
    for backend in BACKENDS:
        if not backend.installed():
            logger.debug("backend %s: not installed (%s)", backend.name, ", ".join(backend.packages))
            continue
        try:
            for module in backend.modules:
                importlib.import_module(f"flashim.backends.{module}")
        except Exception:
            logger.exception("backend %s failed to load; its impls are unavailable", backend.name)
            continue
        LOADED.append(backend.name)
    for entry in entry_points(group="flashim.backends"):
        try:
            loaded = entry.load()
            if callable(loaded):
                loaded()
            LOADED.append(entry.name)
        except Exception:
            logger.exception("backend plugin %r failed to load", entry.name)
    logger.debug("backends loaded: %s", ", ".join(LOADED))
