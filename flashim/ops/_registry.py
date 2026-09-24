"""Op dispatch.

Every op has a torch *reference* implementation (the correctness oracle and the
universal fallback). Optimized implementations register for a
:class:`DeviceSpec` (device type plus an optional compute-capability range) and
may require platform :class:`~flashim.platforms.Feature` s and a per-call
``when`` predicate. Selection order: highest ``priority`` first, then the
narrowest capability range, then registration order.

Runtime overrides (``FLASHIM_OPS`` env var or :func:`configure`), comma separated:
  ``-rope_qk``           disable optimized impls (use the reference)
  ``rope_qk:triton``     force a specific impl
Strict mode (``FLASHIM_STRICT_OPS=1`` or :func:`set_strict`) raises when a
non-CPU call would fall back to the reference.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch

from flashim.platforms import Feature, Platform, platform_for

logger = logging.getLogger(__name__)

REFERENCE = "reference"


@dataclass(frozen=True)
class DeviceSpec:
    type: str
    min_capability: tuple[int, int] | None = None
    max_capability: tuple[int, int] | None = None

    def matches(self, platform: Platform) -> bool:
        if self.type != "*" and self.type != platform.device_type:
            return False
        if self.min_capability is None and self.max_capability is None:
            return True
        if platform.capability is None:
            return False
        if self.min_capability is not None and platform.capability < self.min_capability:
            return False
        if self.max_capability is not None and platform.capability > self.max_capability:
            return False
        return True

    @property
    def span(self) -> float:
        """Width of the capability range; narrower (more specialized) wins ties."""
        if self.type == "*":
            return float("inf")
        low = self.min_capability or (0, 0)
        high = self.max_capability or (99, 99)
        return (high[0] * 100 + high[1]) - (low[0] * 100 + low[1])


ANY = DeviceSpec("*")


@dataclass
class Impl:
    name: str
    fn: Callable
    device: DeviceSpec
    requires: frozenset[Feature] = frozenset()
    priority: int = 0
    when: Callable[..., bool] | None = None
    order: int = 0

    def available(self, platform: Platform) -> bool:
        return self.device.matches(platform) and platform.supports(*self.requires)


@dataclass
class _Config:
    disabled: set[str] = field(default_factory=set)
    forced: dict[str, str] = field(default_factory=dict)
    strict: bool = False


_CONFIG = _Config()
OPS: dict[str, Op] = {}


def configure(spec: str | None) -> None:
    """Apply a ``FLASHIM_OPS``-style override string (replaces previous overrides)."""
    _CONFIG.disabled.clear()
    _CONFIG.forced.clear()
    for token in (spec or "").split(","):
        token = token.strip()
        if not token or token.startswith("+"):
            continue
        if token.startswith("-"):
            _CONFIG.disabled.add(token[1:])
        elif ":" in token:
            op, impl = token.split(":", 1)
            _CONFIG.forced[op] = impl
        else:
            raise ValueError(f"Invalid FLASHIM_OPS token {token!r}")


def set_strict(strict: bool) -> None:
    _CONFIG.strict = strict


configure(os.environ.get("FLASHIM_OPS"))
set_strict(os.environ.get("FLASHIM_STRICT_OPS", "0") == "1")


def _first_device(values: Iterable) -> torch.device | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, (list, tuple)):
            device = _first_device(value)
            if device is not None:
                return device
    return None


class Op:
    def __init__(self, name: str, reference: Callable) -> None:
        if name in OPS:
            raise ValueError(f"Duplicate op {name!r}")
        self.name = name
        self.reference = reference
        self.impls: list[Impl] = []
        self.calls: Counter[str] = Counter()
        self.__doc__ = reference.__doc__
        OPS[name] = self

    def register(
        self,
        name: str,
        *,
        device: DeviceSpec,
        requires: Iterable[Feature] = (),
        priority: int = 0,
        when: Callable[..., bool] | None = None,
    ) -> Callable[[Callable], Callable]:
        if name == REFERENCE or any(impl.name == name for impl in self.impls):
            raise ValueError(f"Duplicate impl {name!r} for op {self.name!r}")

        def decorator(fn: Callable) -> Callable:
            self.impls.append(
                Impl(name, fn, device, frozenset(requires), priority, when, len(self.impls))
            )
            return fn

        return decorator

    def candidates(self, platform: Platform) -> list[Impl]:
        available = [impl for impl in self.impls if impl.available(platform)]
        return sorted(available, key=lambda i: (-i.priority, i.device.span, i.order))

    def select(self, *args, **kwargs) -> tuple[str, Callable]:
        device = _first_device(args) or _first_device(kwargs.values()) or torch.device("cpu")
        platform = platform_for(device)
        forced = _CONFIG.forced.get(self.name)
        if forced == REFERENCE or self.name in _CONFIG.disabled:
            return REFERENCE, self.reference
        if forced is not None:
            for impl in self.impls:
                if impl.name == forced:
                    return impl.name, impl.fn
            raise ValueError(f"Op {self.name!r} has no impl {forced!r}")
        for impl in self.candidates(platform):
            if impl.when is None or impl.when(*args, **kwargs):
                return impl.name, impl.fn
        if _CONFIG.strict and platform.device_type != "cpu":
            raise RuntimeError(
                f"Op {self.name!r} fell back to the reference on {platform.describe()}"
            )
        return REFERENCE, self.reference

    def _count(self, name: str) -> None:
        # Counting inside torch.compile would add guards on the counter values.
        if not torch.compiler.is_compiling():
            self.calls[name] += 1

    def __call__(self, *args, **kwargs):
        name, fn = self.select(*args, **kwargs)
        self._count(name)
        return fn(*args, **kwargs)

    def get(self, name: str | None) -> Callable:
        """Callable pinned to impl ``name`` (``None`` = normal dispatch).

        Call sites that must reproduce a specific numerical path (e.g. an upstream
        model's eager math) pin it here; the global overrides still apply first.
        A pinned impl whose ``when`` predicate rejects the arguments, or that is not
        available on the device, falls back to the reference.
        """
        if name is None:
            return self
        if name == REFERENCE:
            impl = None
        else:
            impl = next((i for i in self.impls if i.name == name), None)
            if impl is None:
                raise ValueError(f"Op {self.name!r} has no impl {name!r}")

        def pinned(*args, **kwargs):
            if self.name in _CONFIG.forced or self.name in _CONFIG.disabled:
                return self(*args, **kwargs)
            device = _first_device(args) or _first_device(kwargs.values()) or torch.device("cpu")
            if impl is not None and impl.available(platform_for(device)) and (
                impl.when is None or impl.when(*args, **kwargs)
            ):
                self._count(impl.name)
                return impl.fn(*args, **kwargs)
            self._count(REFERENCE)
            return self.reference(*args, **kwargs)

        return pinned

    def __repr__(self) -> str:
        impls = ", ".join(i.name for i in self.impls)
        return f"Op({self.name}: reference, {impls})"


def reset_call_counts() -> None:
    for op in OPS.values():
        op.calls.clear()


def call_counts() -> dict[str, dict[str, int]]:
    return {name: dict(op.calls) for name, op in OPS.items() if op.calls}
