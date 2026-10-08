# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/graph_compile.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Opt-in ``torch.compile`` adapters for RoMaV2 inference hot regions.

The adapter compiles the target modules' bound ``forward`` methods rather than
replacing registered modules.  Keeping the original module objects in place
preserves parameter names and therefore the model's ``state_dict`` contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "GraphCompileReport",
    "GraphCompileRequest",
    "compile_graph_regions_for_inference",
]


_COMPILE_MARKER = "_romav2_graph_regions_compiled"
_TARGET_REGION_NAMES = ("f", "matcher.mv_vit")


@dataclass(frozen=True)
class GraphCompileRequest:
    """Explicit selection and options for graph-region compilation.

    ``targets`` may contain any unique subset of ``"f"`` and
    ``"matcher.mv_vit"``. It defaults to both regions, matching the legacy
    invocation that does not pass a request.
    """

    targets: tuple[str, ...] = _TARGET_REGION_NAMES
    dynamic: bool = False
    mode: str | None = None


@dataclass(frozen=True)
class GraphCompileReport:
    """Result of an inference graph-region compilation request.

    ``selected`` contains regions compiled by this call. ``skipped`` contains
    known target-region names that were absent, unsuitable, or already compiled.
    """

    selected: tuple[str, ...]
    skipped: tuple[str, ...]
    dynamic: bool
    mode: str | None


def _validate_target_names(targets: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(targets, tuple):
        raise TypeError("compile request targets must be a tuple of target names")
    if not all(isinstance(name, str) for name in targets):
        raise TypeError("compile request targets must contain only strings")

    duplicate_names = tuple(
        name for index, name in enumerate(targets) if name in targets[:index]
    )
    if duplicate_names:
        raise ValueError(f"duplicate graph compile targets: {duplicate_names!r}")

    unknown_names = tuple(name for name in targets if name not in _TARGET_REGION_NAMES)
    if unknown_names:
        raise ValueError(f"unknown graph compile targets: {unknown_names!r}")
    return targets


def _target_regions(model: nn.Module) -> tuple[tuple[str, object], ...]:
    matcher = getattr(model, "matcher", None)
    matcher_vit = getattr(matcher, "mv_vit", None) if matcher is not None else None
    return (("f", getattr(model, "f", None)), ("matcher.mv_vit", matcher_vit))

def _resolve_torch_compile() -> Callable[..., Callable[..., object]] | None:
    """Return the installed PyTorch graph compiler, when available."""

    return getattr(torch, "compile", None)


def _is_already_compiled(region: nn.Module) -> bool:
    """Recognize both this adapter's forwards and torch's module wrapper."""

    return bool(region.__dict__.get(_COMPILE_MARKER, False)) or hasattr(
        region, "_orig_mod"
    )


def _clone_tensor_leaves(value: object) -> object:
    """Clone tensors in compiled results before CUDA Graph storage can be reused."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_tensor_leaves(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_leaves(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tensor_leaves(item) for key, item in value.items()}
    return value


def _compile_region(
    region: nn.Module,
    *,
    dynamic: bool,
    mode: str | None,
    compiler: Callable[..., Callable[..., object]],
) -> None:
    # Compile a bound method and keep the module registered at its original path.
    # Replacing the module with OptimizedModule would prefix state-dict keys with
    # ``_orig_mod`` and make existing checkpoints needlessly incompatible.
    compiled_forward = compiler(
        region.forward,
        fullgraph=False,
        dynamic=dynamic,
        mode=mode,
    )
    if mode == "reduce-overhead":
        mark_step_begin = getattr(
            getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
        )

        def clone_compiled_result(*args: object, **kwargs: object) -> object:
            if callable(mark_step_begin):
                mark_step_begin()
            return _clone_tensor_leaves(compiled_forward(*args, **kwargs))

        region.forward = clone_compiled_result  # type: ignore[method-assign]
    else:
        region.forward = compiled_forward  # type: ignore[method-assign]
    region.__dict__[_COMPILE_MARKER] = True


def compile_graph_regions_for_inference(
    model: nn.Module,
    *,
    request: GraphCompileRequest | None = None,
    dynamic: bool = False,
    mode: str | None = None,
) -> GraphCompileReport:
    """Compile selected descriptor and matcher-transformer regions for inference.

    The caller opts into compilation by invoking this function, including for
    CPU-resident modules. With no ``request``, it preserves the legacy behavior:
    it considers both ``model.f`` and ``model.matcher.mv_vit`` and takes options
    from ``dynamic`` and ``mode``. A :class:`GraphCompileRequest` selects a unique
    subset of those documented regions and carries its own options. Refiner paths
    and ``match()`` stay untouched. ``fullgraph=False`` deliberately permits graph
    breaks for custom operations that Dynamo cannot capture.

    The entire model and each requested region must be in evaluation mode. Target
    names are validated before any region forward is replaced.

    Raises:
        TypeError: If ``model`` is not an :class:`~torch.nn.Module`, ``request`` is
            not a :class:`GraphCompileRequest`, or its targets are malformed.
        ValueError: If a request repeats or does not name a documented target, or
            if request options are combined with legacy option keywords.
        RuntimeError: If evaluation mode is not active or ``torch.compile`` is
            unavailable in the installed PyTorch build.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")

    if request is None:
        effective_request = GraphCompileRequest(dynamic=dynamic, mode=mode)
    else:
        if not isinstance(request, GraphCompileRequest):
            raise TypeError("request must be a GraphCompileRequest")
        if dynamic or mode is not None:
            raise ValueError(
                "request options cannot be combined with dynamic or mode keywords"
            )
        effective_request = request
    target_names = _validate_target_names(effective_request.targets)

    if model.training:
        raise RuntimeError("graph-region compilation requires model.eval()")

    compiler = _resolve_torch_compile()
    if not callable(compiler):
        raise RuntimeError(  # noqa: TRY004
            "torch.compile is unavailable; install a PyTorch build with Dynamo support"
        )

    selected: list[str] = []
    skipped: list[str] = []
    eligible: list[tuple[str, nn.Module]] = []
    regions_by_name = dict(_target_regions(model))
    for name in target_names:
        region = regions_by_name[name]
        if (
            not isinstance(region, nn.Module)
            or not callable(region)
            or _is_already_compiled(region)
        ):
            skipped.append(name)
        elif region.training:
            raise RuntimeError(
                f"graph-region compilation requires {name} to be in eval mode"
            )
        else:
            eligible.append((name, region))

    for name, region in eligible:
        _compile_region(
            region,
            dynamic=effective_request.dynamic,
            mode=effective_request.mode,
            compiler=compiler,
        )
        selected.append(name)

    return GraphCompileReport(
        selected=tuple(selected),
        skipped=tuple(skipped),
        dynamic=effective_request.dynamic,
        mode=effective_request.mode,
    )
