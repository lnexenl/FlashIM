"""FlashIM logging: stdlib :mod:`logging` under the ``flashim`` namespace.

The library only emits records; it installs no handler unless asked. Opt in with the ``FLASHIM_LOG`` env var or
:func:`set_verbosity`. ``FLASHIM_LOG`` is a comma-separated list of levels, optionally per subsystem (the
``flashim.`` prefix may be omitted)::

    FLASHIM_LOG=info                 # everything at INFO
    FLASHIM_LOG=info,ops=debug       # plus op dispatch decisions (first use of each impl, rejection reasons)

Subsystems: ``flashim.ops`` (dispatch), ``flashim.ops._cuda`` (JIT builds), ``flashim.platforms``,
``flashim.quant``, ``flashim.models.<name>``.

Nothing here may run inside ``torch.compile`` tracing (a logger call there graph-breaks or is dropped): callers
guard with ``torch.compiler.is_compiling()``, and :func:`warn_once` does so itself.

Profiler ranges: op calls are wrapped in ``flashim::<op>[<impl>]`` ranges while a ``torch.profiler`` session is
active, or as NVTX ranges (for nsys / ncu ``--nvtx``) when ``FLASHIM_NVTX=1``. Eager only; compiled regions show
Inductor's kernel names.
"""

from __future__ import annotations

import contextlib
import logging
import os

import torch

ROOT = "flashim"
_FORMAT = "%(asctime)s %(levelname).1s %(name)s: %(message)s"
_handler: logging.Handler | None = None
_warned: set[str] = set()
NVTX = os.environ.get("FLASHIM_NVTX", "0") == "1"

logging.getLogger(ROOT).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Logger for ``name`` (a module ``__name__`` or a subsystem such as ``"ops"``), always under ``flashim``."""
    return logging.getLogger(name if name == ROOT or name.startswith(ROOT + ".") else f"{ROOT}.{name}")


def _parse_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    levels = logging.getLevelNamesMapping()
    if level.strip().upper() not in levels:
        raise ValueError(f"Unknown log level {level!r}; choose from {sorted(levels)}")
    return levels[level.strip().upper()]


def set_verbosity(spec: str | int, *, handler: bool = True) -> None:
    """Set levels from a ``FLASHIM_LOG``-style spec (or one level) and, unless ``handler=False`` (the application
    configures logging itself), attach a stderr handler to the ``flashim`` logger."""
    global _handler
    tokens = [spec] if isinstance(spec, int) else [t for t in spec.split(",") if t.strip()]
    for token in tokens:
        if isinstance(token, str) and "=" in token:
            name, level = token.split("=", 1)
            get_logger(name.strip()).setLevel(_parse_level(level))
        else:
            logging.getLogger(ROOT).setLevel(_parse_level(token))
    if handler and _handler is None:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        logging.getLogger(ROOT).addHandler(_handler)


def warn_once(logger: logging.Logger, key: str, message: str, *args, level: int = logging.WARNING) -> None:
    """Log ``message`` the first time ``key`` is seen (no-op while ``torch.compile`` traces)."""
    if torch.compiler.is_compiling() or key in _warned:
        return
    _warned.add(key)
    logger.log(level, message, *args)


def profiler_range(name: str):
    """A ``torch.profiler`` / NVTX range named ``name`` when one is being recorded, else a null context."""
    if NVTX and torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    if torch.autograd._profiler_enabled():
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


if os.environ.get("FLASHIM_LOG"):
    set_verbosity(os.environ["FLASHIM_LOG"])
