"""Op dispatch.

Every op has a torch *reference* implementation (the correctness oracle and the
universal fallback). Optimized implementations register for a
:class:`DeviceSpec` (a platform dispatch key -- ``"cuda"`` is NVIDIA only, ROCm is
``"rocm"`` -- plus an optional NVIDIA compute-capability range) and may require
platform :class:`~flashim.platforms.Feature` s and a per-call ``when`` predicate.
Require a feature rather than a capability range when the hardware feature is not
monotonic in the capability (``Feature.WGMMA``, ``Feature.TCGEN05``).
Selection order: highest ``priority`` first, then the narrowest capability range,
then registration order. :meth:`Op.explain` says why each impl was (not) chosen;
ops with constant weight operands (``Op(..., weights=n)``) let each impl repack them
once, at load time, through :meth:`Op.bind` (``register(..., prepare=...)``);
``FLASHIM_LOG=ops=debug`` logs the first use of every impl and reference fallbacks
on accelerators (see :mod:`flashim.log`).

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
from dataclasses import dataclass, field, fields, is_dataclass

import torch

from flashim.log import profiler_range, warn_once
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
        return self.max_capability is None or platform.capability <= self.max_capability

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
    # Raw weight operands -> this impl's layout (tuple), run once per BoundOp (per call when unbound).
    prepare: Callable[..., tuple] | None = None

    def available(self, platform: Platform) -> bool:
        return self.device.matches(platform) and platform.supports(*self.requires)

    def rejection(self, platform: Platform, args: tuple, kwargs: dict) -> str | None:
        """Why this impl does not apply to the call (``None`` if it does)."""
        if not self.device.matches(platform):
            return f"device {self.device} does not match {platform.device_type} {platform.arch}".rstrip()
        missing = sorted(f.value for f in self.requires if not platform.supports(f))
        if missing:
            return "missing features " + ",".join(missing)
        if self.when is not None and not self.when(*args, **kwargs):
            return "when() rejected the arguments"
        return None


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
        if is_dataclass(value) and not isinstance(value, type):  # QuantizedTensor / QuantizedWeight etc.
            value = [getattr(value, f.name) for f in fields(value)]
        if isinstance(value, (list, tuple)):
            device = _first_device(value)
            if device is not None:
                return device
    return None


class Op:
    """A dispatched op. ``composite=True`` marks a reference built from other (dispatched) ops -- a fused op's
    unfused pipeline -- so falling back to it is expected and not reported as a slow path.

    ``weights=n``: the last ``n`` positional arguments are constant weights (raw module parameters). ``when``
    predicates and the reference see the raw weights; an impl registered with ``prepare`` receives
    ``prepare(*weights)`` in their place. Use :meth:`bind` to prepare once; a plain call prepares every time.
    """

    def __init__(self, name: str, reference: Callable, *, composite: bool = False, weights: int = 0) -> None:
        if name in OPS:
            raise ValueError(f"Duplicate op {name!r}")
        self.name = name
        self.reference = reference
        self.composite = composite
        self.weights = weights
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
        prepare: Callable[..., tuple] | None = None,
    ) -> Callable[[Callable], Callable]:
        if name == REFERENCE or any(impl.name == name for impl in self.impls):
            raise ValueError(f"Duplicate impl {name!r} for op {self.name!r}")
        if prepare is not None and not self.weights:
            raise ValueError(f"Op {self.name!r} declares no weight operands to prepare")

        def decorator(fn: Callable) -> Callable:
            self.impls.append(
                Impl(name, fn, device, frozenset(requires), priority, when, len(self.impls), prepare)
            )
            return fn

        return decorator

    def candidates(self, platform: Platform) -> list[Impl]:
        available = [impl for impl in self.impls if impl.available(platform)]
        return sorted(available, key=lambda i: (-i.priority, i.device.span, i.order))

    def _unknown_impl(self, name: str) -> str:
        from flashim.backends import BACKENDS, LOADED

        missing = [f"{b.name} (needs {', '.join(b.packages)})" for b in BACKENDS if b.name not in LOADED]
        hint = f"; backends not loaded: {', '.join(missing)}" if missing else ""
        return f"Op {self.name!r} has no impl {name!r} (registered: {[i.name for i in self.impls]}){hint}"

    def _platform(self, args: tuple, kwargs: dict) -> Platform:
        return platform_for(_first_device(args) or _first_device(kwargs.values()) or torch.device("cpu"))

    def explain(self, *args, **kwargs) -> dict[str, str]:
        """``{impl: "selected" | why not}`` for a call with these arguments, in selection order."""
        platform = self._platform(args, kwargs)
        chosen, _ = self.select(*args, **kwargs)
        ranked = self.candidates(platform)
        ranked += [impl for impl in self.impls if impl not in ranked]
        report = {}
        for impl in ranked:
            if impl.name == chosen:
                report[impl.name] = "selected"
            else:
                report[impl.name] = impl.rejection(platform, args, kwargs) or "lower priority"
        report[REFERENCE] = "selected" if chosen == REFERENCE else "fallback"
        return report

    def _find(self, name: str) -> Impl:
        impl = next((i for i in self.impls if i.name == name), None)
        if impl is None:
            raise ValueError(self._unknown_impl(name))
        return impl

    def _choose(self, args: tuple, kwargs: dict, pin: str | None = None) -> Impl | None:
        """The impl to run (``None`` = the reference). Global overrides first; a pinned impl that does not apply
        falls back to the reference, not to other impls."""
        forced = _CONFIG.forced.get(self.name)
        if forced == REFERENCE or self.name in _CONFIG.disabled:
            return None
        if forced is not None:
            return self._find(forced)
        if pin == REFERENCE:
            return None
        platform = self._platform(args, kwargs)
        for impl in [self._find(pin)] if pin is not None else self.candidates(platform):
            if impl.available(platform) and (impl.when is None or impl.when(*args, **kwargs)):
                return impl
        if pin is None and _CONFIG.strict and platform.device_type != "cpu":
            raise RuntimeError(f"Op {self.name!r} fell back to the reference on {platform.describe()}")
        return None

    def select(self, *args, **kwargs) -> tuple[str, Callable]:
        impl = self._choose(args, kwargs)
        return (REFERENCE, self.reference) if impl is None else (impl.name, impl.fn)

    def _invoke(
        self, impl: Impl | None, args: tuple, kwargs: dict, pin: str | None = None, prepared: tuple | None = None
    ):
        pinned = pin is not None and (impl is not None or pin == REFERENCE)
        if impl is None:
            return self._run(REFERENCE, self.reference, args, kwargs, pinned)
        call_args = args
        if impl.prepare is not None:
            split = len(args) - self.weights
            if prepared is None:
                prepared = tuple(impl.prepare(*args[split:]))
            call_args = args[:split] + prepared
        return self._run(impl.name, impl.fn, args, kwargs, pinned, call_args)

    def _run(
        self, name: str, fn: Callable, args: tuple, kwargs: dict, pinned: bool = False, call_args: tuple | None = None
    ):
        call_args = args if call_args is None else call_args
        # Counting / logging inside torch.compile would add guards or graph breaks: eager only.
        if torch.compiler.is_compiling():
            return fn(*call_args, **kwargs)
        if name not in self.calls:
            self._log_first_use(name, args, kwargs, pinned)
        self.calls[name] += 1
        with profiler_range(f"flashim::{self.name}[{name}]"):
            return fn(*call_args, **kwargs)

    def _log_first_use(self, name: str, args: tuple, kwargs: dict, pinned: bool) -> None:
        if pinned or self.composite or name != REFERENCE or not self.impls:
            logger.debug("op %s -> %s%s", self.name, name, " (pinned)" if pinned else "")
            return
        platform = self._platform(args, kwargs)
        if platform.device_type == "cpu" or not logger.isEnabledFor(logging.INFO):
            return
        reasons = "; ".join(
            f"{impl.name}: {impl.rejection(platform, args, kwargs) or 'forced off'}" for impl in self.impls
        )
        warn_once(
            logger, f"reference:{self.name}:{platform.device_type}", "op %s uses the reference on %s (%s)",
            self.name, platform.describe(), reasons, level=logging.INFO,
        )

    def __call__(self, *args, **kwargs):
        return self._invoke(self._choose(args, kwargs), args, kwargs)

    def bind(self, *weights, impl: str | None = None) -> BoundOp:
        """This op with its ``weights`` trailing operands fixed (see :class:`BoundOp`); ``impl`` pins an impl
        like :meth:`get`."""
        if len(weights) != self.weights:
            raise ValueError(f"Op {self.name!r} takes {self.weights} weight operands, got {len(weights)}")
        if impl is not None and impl != REFERENCE:
            self._find(impl)
        return BoundOp(self, weights, impl)

    def get(self, name: str | None) -> Callable:
        """Callable pinned to impl ``name`` (``None`` = normal dispatch).

        Call sites that must reproduce a specific numerical path (e.g. an upstream
        model's eager math) pin it here; the global overrides still apply first.
        A pinned impl whose ``when`` predicate rejects the arguments, or that is not
        available on the device, falls back to the reference.
        """
        if name is None:
            return self
        if name != REFERENCE:
            self._find(name)

        def pinned(*args, **kwargs):
            return self._invoke(self._choose(args, kwargs, pin=name), args, kwargs, pin=name)

        return pinned

    def __repr__(self) -> str:
        impls = ", ".join(i.name for i in self.impls)
        return f"Op({self.name}: reference, {impls})"


class BoundOp:
    """An op whose trailing weight operands are fixed (a layer's parameters, bound once at load time).

    Calls pass only the leading (activation) operands. Dispatch is unchanged -- priority, ``when`` on the full
    arguments, overrides, pinning -- but each impl's ``prepare`` runs once, the first time that impl is chosen, and
    its packed weights are reused. Rebind after changing the weights in place.
    """

    def __init__(self, op: Op, weights: tuple, pin: str | None = None) -> None:
        self.op = op
        self.weights = weights
        self.pin = pin
        self._prepared: dict[str, tuple] = {}

    @property
    def platform(self) -> Platform:
        return platform_for(_first_device(self.weights) or torch.device("cpu"))

    @property
    def available(self) -> list[str]:
        """Impls that can run on the weights' device (before per-call ``when`` checks), best first."""
        names = [impl.name for impl in self.op.candidates(self.platform)]
        return names if self.pin is None else [n for n in names if n == self.pin]

    @property
    def primary(self) -> str:
        """The impl calls will use when their arguments qualify."""
        return next(iter(self.available), REFERENCE)

    def __call__(self, *args, **kwargs):
        full = args + self.weights
        impl = self.op._choose(full, kwargs, pin=self.pin)
        prepared = None
        if impl is not None and impl.prepare is not None:
            prepared = self._prepared.get(impl.name)
            if prepared is None:
                prepared = self._prepared[impl.name] = tuple(impl.prepare(*self.weights))
        return self.op._invoke(impl, full, kwargs, pin=self.pin, prepared=prepared)

    def __repr__(self) -> str:
        return f"BoundOp({self.op.name} -> {self.primary})"


def reset_call_counts() -> None:
    for op in OPS.values():
        op.calls.clear()


def call_counts() -> dict[str, dict[str, int]]:
    return {name: dict(op.calls) for name, op in OPS.items() if op.calls}
