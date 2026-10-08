# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/fast_fp8_graph.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""CUDA-graph capture and replay for fixed-shape FastFP8 forward execution.

The static FP8 runtime (:mod:`romav2.fast_fp8`) guarantees a fixed sequence of
quantize → FP8 GEMM → dequantize kernels per forward with no per-call module
discovery, packing, fingerprinting, or dynamic quantization. That makes the
forward pass a fixed-shape workload suitable for whole-graph capture. This
module wraps any CUDA ``nn.Module`` — in practice a
:class:`romav2.fast_fp8.FastFP8Model` — in :class:`FastFP8CudaGraph`, which:

1. **Preallocates static inputs** — every tensor leaf of the example inputs
   given to :meth:`FastFP8CudaGraph.capture` is cloned once into a persistent
   static tensor; the model is warmed up and captured reading exactly those
   tensors.
2. **Records one CUDA graph** — the forward is captured into a
   ``torch.cuda.CUDAGraph`` backed by a dedicated graph memory pool (or a
   caller-supplied pool for sharing across wrappers).
3. **Replays without Python or allocation** — :meth:`FastFP8CudaGraph.replay`
   validates the caller's inputs against the captured structure (exact
   nesting, shapes, dtypes, and devices), copies them into the static input
   tensors, replays the graph, and returns the **graph-owned output tensors**
   retained at capture time. Replay never clones, never allocates output
   storage, and never re-enters the model's Python ``forward`` — the same
   output objects are returned every call, with values updated in place by
   the graph replay.

Scope is strictly fixed-shape tensor forward execution after image
preprocessing: there is no file IO, no path-based ``match`` capture, and no
generic dispatch. CUDA graphs require a CUDA device — construction raises an
explicit :class:`RuntimeError` on CPU-only hosts, capture raises for
non-CUDA inputs and preflights ``model.to(device)`` so CPU-resident or
mixed-device models are migrated rather than rejected, and every replay
validation failure raises :class:`RuntimeError` naming the offending aspect
(structure, shape, dtype, or device).

Workspace accounting
--------------------
``workspace_bytes`` is 0 before capture. After capture it reports the total
bytes pinned by the wrapper: the preallocated static input tensors plus the
growth in reserved caching-allocator memory attributable to the graph's
private memory pool (captured activations, cuBLAS workspaces, and the
graph-owned outputs). It is recorded once at capture and does not change
across replays.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["FastFP8CudaGraph"]

# Input-tree spec tags. A spec mirrors the captured input structure; leaves
# are ("tensor", shape, dtype, device) or ("const", value), containers are
# ("tuple", children) / ("namedtuple", cls, children) / ("list", children) /
# ("dict", keys, children).
_SPEC_TENSOR = "tensor"
_SPEC_TUPLE = "tuple"
_SPEC_NAMEDTUPLE = "namedtuple"
_SPEC_LIST = "list"
_SPEC_DICT = "dict"
_SPEC_CONST = "const"

_CONSTANT_LEAF_TYPES = (bool, int, float, complex, str, bytes, type(None))


def _freeze_input(node: Any, path: str, devices: list[torch.device]) -> tuple[tuple, Any]:
    """Build a (spec, static) pair for one captured input node.

    Tensor leaves are validated as CUDA tensors and cloned into persistent
    static tensors; ``devices`` collects every tensor device seen so capture
    can enforce a single target device. Container nodes recurse; anything
    else must be a supported constant leaf and is baked into the graph.
    """
    if isinstance(node, Tensor):
        if node.device.type != "cuda":
            raise RuntimeError(
                "FastFP8CudaGraph capture requires CUDA tensor inputs; "
                f"{path} is a tensor on device {node.device}"
            )
        devices.append(node.device)
        spec = (_SPEC_TENSOR, tuple(node.shape), node.dtype, node.device)
        return spec, node.clone()
    if isinstance(node, tuple) and hasattr(node, "_fields"):
        cls = type(node)
        specs, statics = _freeze_sequence(node, path, devices)
        return (_SPEC_NAMEDTUPLE, cls, specs), cls(*statics)
    if isinstance(node, tuple):
        specs, statics = _freeze_sequence(node, path, devices)
        return (_SPEC_TUPLE, specs), tuple(statics)
    if isinstance(node, list):
        specs, statics = _freeze_sequence(node, path, devices)
        return (_SPEC_LIST, specs), statics
    if isinstance(node, dict):
        keys: list[Any] = []
        specs: list[tuple] = []
        static: dict[Any, Any] = {}
        for key, value in node.items():
            child_spec, child_static = _freeze_input(value, f"{path}[{key!r}]", devices)
            keys.append(key)
            specs.append(child_spec)
            static[key] = child_static
        return (_SPEC_DICT, tuple(keys), tuple(specs)), static
    if isinstance(node, _CONSTANT_LEAF_TYPES):
        return (_SPEC_CONST, node), node
    raise RuntimeError(
        "FastFP8CudaGraph capture does not support input of type "
        f"{type(node).__name__} at {path}; expected a tensor, tuple, list, "
        "dict, or constant scalar leaf"
    )


def _freeze_sequence(
    nodes: Any, path: str, devices: list[torch.device]
) -> tuple[tuple[tuple, ...], list[Any]]:
    specs: list[tuple] = []
    statics: list[Any] = []
    for index, item in enumerate(nodes):
        spec, static = _freeze_input(item, f"{path}[{index}]", devices)
        specs.append(spec)
        statics.append(static)
    return tuple(specs), statics


def _validate_input(spec: tuple, node: Any, path: str) -> None:
    """Validate a replay input against its captured spec.

    Every failure raises :class:`RuntimeError` naming the failing aspect so
    callers can match loosely on "structure" / "shape" / "dtype" / "device".
    """
    tag = spec[0]
    if tag == _SPEC_TENSOR:
        _, shape, dtype, device = spec
        if not isinstance(node, Tensor):
            raise RuntimeError(
                f"FastFP8CudaGraph replay input structure mismatch at {path}: "
                f"expected a tensor of shape {shape}, dtype {dtype}, device "
                f"{device}, got {type(node).__name__}"
            )
        if tuple(node.shape) != shape:
            raise RuntimeError(
                f"FastFP8CudaGraph replay input shape mismatch at {path}: "
                f"captured {shape}, got {tuple(node.shape)}"
            )
        if node.dtype != dtype:
            raise RuntimeError(
                f"FastFP8CudaGraph replay input dtype mismatch at {path}: "
                f"captured {dtype}, got {node.dtype}"
            )
        if node.device != device:
            raise RuntimeError(
                f"FastFP8CudaGraph replay input device mismatch at {path}: "
                f"captured {device}, got {node.device}"
            )
        return
    if tag in (_SPEC_TUPLE, _SPEC_NAMEDTUPLE, _SPEC_LIST):
        if tag == _SPEC_LIST:
            container_type: type = list
            children = spec[1]
            container_name = "list"
        elif tag == _SPEC_TUPLE:
            container_type = tuple
            children = spec[1]
            container_name = "tuple"
        else:
            container_type = spec[1]
            children = spec[2]
            container_name = spec[1].__name__
        got = (
            f"{type(node).__name__} of {len(node)} element(s)"
            if isinstance(node, (tuple, list))
            else type(node).__name__
        )
        if type(node) is not container_type or len(node) != len(children):
            raise RuntimeError(
                f"FastFP8CudaGraph replay input structure mismatch at {path}: "
                f"expected {container_name} of {len(children)} element(s), got {got}"
            )
        for index, (child_spec, value) in enumerate(zip(children, node)):
            _validate_input(child_spec, value, f"{path}[{index}]")
        return
    if tag == _SPEC_DICT:
        _, keys, children = spec
        if type(node) is not dict:
            raise RuntimeError(
                f"FastFP8CudaGraph replay input structure mismatch at {path}: "
                f"expected dict with keys {[repr(key) for key in keys]}, got "
                f"{type(node).__name__}"
            )
        captured_keys = set(keys)
        if set(node) != captured_keys:
            missing = [repr(key) for key in keys if key not in node]
            unexpected = [repr(key) for key in node if key not in captured_keys]
            raise RuntimeError(
                f"FastFP8CudaGraph replay input structure mismatch at {path}: "
                f"expected dict keys {[repr(key) for key in keys]}; missing "
                f"{missing}, unexpected {unexpected}"
            )
        for key, child_spec in zip(keys, children):
            _validate_input(child_spec, node[key], f"{path}[{key!r}]")
        return
    # Constant leaf: baked into the graph; replay must pass the same value.
    if type(node) is not type(spec[1]) or node != spec[1]:
        raise RuntimeError(
            f"FastFP8CudaGraph replay input structure mismatch at {path}: "
            f"expected constant {spec[1]!r}, got {node!r}"
        )


def _copy_into_static(static: Any, node: Any) -> None:
    """Copy a validated replay input into the preallocated static tensors."""
    if isinstance(static, Tensor):
        static.copy_(node, non_blocking=True)
    elif isinstance(static, dict):
        for key, value in static.items():
            _copy_into_static(value, node[key])
    elif isinstance(static, (tuple, list)):
        for target, value in zip(static, node):
            _copy_into_static(target, value)
    # Constant leaves carry no storage; nothing to copy.


def _static_tensor_bytes(tree: Any) -> int:
    """Sum the allocated storage bytes of every tensor in a static tree."""
    if isinstance(tree, Tensor):
        return tree.untyped_storage().nbytes()
    if isinstance(tree, dict):
        return sum(_static_tensor_bytes(value) for value in tree.values())
    if isinstance(tree, (tuple, list)):
        return sum(_static_tensor_bytes(value) for value in tree)
    return 0


class _CaptureMethod:
    """Dual-binding ``capture`` for :class:`FastFP8CudaGraph`.

    Instance access — ``wrapper.capture(*args, **kwargs)`` — captures the
    wrapper's model and returns the wrapper (chainable). Class access —
    ``FastFP8CudaGraph.capture(model, *args, **kwargs)`` — constructs a
    wrapper for ``model`` first, matching the ``FastFP8Model.capture_graph``
    convenience call. Both forms accept the keyword-only options
    ``warmup_iters`` and ``pool``; every other keyword is forwarded to the
    model.
    """

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        owner = objtype if objtype is not None else FastFP8CudaGraph
        if obj is None:

            def capture(
                model: nn.Module,
                *args: Any,
                warmup_iters: int = 3,
                pool: Any = None,
                **kwargs: Any,
            ) -> FastFP8CudaGraph:
                """Construct a wrapper for ``model`` and capture its forward."""
                return owner(model)._capture(
                    args, kwargs, warmup_iters=warmup_iters, pool=pool
                )

            return capture

        def capture(
            *args: Any,
            warmup_iters: int = 3,
            pool: Any = None,
            **kwargs: Any,
        ) -> FastFP8CudaGraph:
            """Capture the wrapped model's forward for these example inputs."""
            return obj._capture(args, kwargs, warmup_iters=warmup_iters, pool=pool)

        return capture


class FastFP8CudaGraph:
    """CUDA-graph capture/replay wrapper for a fixed-shape CUDA model forward.

    Lifecycle: construct around a CUDA ``nn.Module``, call
    :meth:`capture` once with representative example inputs, then call
    :meth:`replay` with fresh inputs of the exact same structure, shapes,
    dtypes, and devices. ``capture`` is also usable unbound as
    ``FastFP8CudaGraph.capture(model, *args, **kwargs)`` (see
    :class:`_CaptureMethod`).

    Facts exposed: :attr:`captured` (bool), :attr:`workspace_bytes` (bytes
    pinned for static inputs plus the graph pool, 0 before capture and
    constant across replays), :attr:`pool` (the graph memory pool handle,
    shareable with other wrappers via ``capture(..., pool=...)``), and
    :attr:`device` (the single CUDA device of the captured inputs).

    Capture permanently moves the wrapped model onto the capture device via
    ``model.to(device)`` (values and dtypes unchanged): synchronizing
    host-to-device copies cannot be recorded into a CUDA graph, and the fast
    FP8 runtime uploads its CPU-resident scale exponents inside ``forward``.

    Replay never executes Python model code: module-level counters such as
    ``FastFP8Model.last_dispatch_stats`` reflect the capture call, not
    replays. Outputs are the graph-owned tensors recorded during capture;
    their storage is rewritten in place on every replay.
    """

    capture = _CaptureMethod()

    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                "FastFP8CudaGraph expects an nn.Module, got "
                f"{type(model).__name__}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FastFP8CudaGraph requires CUDA, but torch.cuda.is_available() "
                "is False"
            )
        # The model's own device placement is deliberately NOT gated here:
        # capture preflights ``model.to(device)`` once the input device is
        # known, so CPU-resident or mixed-device models are migrated at
        # capture time instead of rejected at construction time.
        self._model = model
        self._captured = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._pool: Any = None
        self._device: torch.device | None = None
        self._workspace_bytes = 0
        self._static_args: tuple[Any, ...] = ()
        self._static_kwargs: dict[str, Any] = {}
        self._args_spec: tuple | None = None
        self._kwargs_spec: tuple | None = None
        self._static_out: Any = None

    @property
    def model(self) -> nn.Module:
        """The wrapped model."""
        return self._model

    @property
    def captured(self) -> bool:
        """Whether :meth:`capture` has completed."""
        return self._captured

    @property
    def workspace_bytes(self) -> int:
        """Bytes pinned by static inputs and the graph memory pool (0 before capture)."""
        return self._workspace_bytes

    @property
    def pool(self) -> Any:
        """The graph memory pool handle, or None before capture.

        Pass this handle to another wrapper's ``capture(..., pool=...)`` to
        share pool memory across graphs. Shared pools are only safe when the
        graphs are replayed in the order they were captured and never
        concurrently.
        """
        return self._pool

    @property
    def device(self) -> torch.device | None:
        """The CUDA device of the captured inputs, or None before capture."""
        return self._device

    def _capture(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        warmup_iters: int,
        pool: Any,
    ) -> FastFP8CudaGraph:
        if self._captured:
            raise RuntimeError(
                "FastFP8CudaGraph has already captured a graph; create a new "
                "wrapper to re-capture"
            )
        if warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")

        devices: list[torch.device] = []
        args_spec, static_args = _freeze_input(tuple(args), "args", devices)
        kwargs_spec, static_kwargs = _freeze_input(kwargs, "kwargs", devices)
        if not devices:
            raise RuntimeError(
                "FastFP8CudaGraph capture requires at least one CUDA tensor input"
            )
        unique_devices = sorted({str(device) for device in devices})
        if len(unique_devices) > 1:
            raise RuntimeError(
                "FastFP8CudaGraph capture requires all tensor inputs on a "
                f"single CUDA device; found {', '.join(unique_devices)}"
            )
        device = devices[0]
        static_input_bytes = _static_tensor_bytes(static_args) + _static_tensor_bytes(
            static_kwargs
        )
        # Preflight: move the whole model onto the capture device (in-place
        # mutation of the caller's model). CUDA graph capture cannot record
        # synchronizing host-to-device copies, which is exactly how the fast
        # FP8 runtime uploads its CPU-resident scalar scale exponents inside
        # ``forward`` (e.g. ``quantize_block32``); after ``Module.to`` those
        # uploads are no-op ``Tensor.to`` calls. The move is permanent —
        # captured kernels reference the migrated storage — and leaves eager
        # results unchanged (and marginally faster, since the per-call H2D
        # copy disappears).
        self._model.to(device)

        with torch.cuda.device(device), torch.no_grad():
            # Warm up on a side stream (per the CUDA-graphs recipe) so lazy
            # initializations (cuBLAS handles/workspaces, kernel autotuning)
            # happen outside capture.
            side_stream = torch.cuda.Stream(device)
            side_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side_stream):
                for _ in range(warmup_iters):
                    self._model(*static_args, **static_kwargs)
            torch.cuda.current_stream(device).wait_stream(side_stream)
            torch.cuda.synchronize(device)

            self._pool = pool if pool is not None else torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                # The graph context frees cached blocks on entry; measure the
                # reserved-memory baseline here so the delta reflects exactly
                # this graph's pool segments (activations, workspaces, and the
                # graph-owned outputs below).
                reserved_before = torch.cuda.memory_reserved(device)
                static_out = self._model(*static_args, **static_kwargs)
            reserved_after = torch.cuda.memory_reserved(device)
            # Stream capture records kernels without executing them; replay
            # once so the retained outputs hold real values immediately.
            graph.replay()
            torch.cuda.synchronize(device)

        self._graph = graph
        self._device = device
        self._args_spec = args_spec
        self._kwargs_spec = kwargs_spec
        self._static_args = static_args
        self._static_kwargs = static_kwargs
        self._static_out = static_out
        self._workspace_bytes = static_input_bytes + max(
            0, reserved_after - reserved_before
        )
        self._captured = True
        return self

    def replay(self, *args: Any, **kwargs: Any) -> Any:
        """Replay the captured graph with new input values.

        ``args``/``kwargs`` must match the captured inputs exactly in nested
        structure, tensor shapes, dtypes, and devices; constant leaves must
        repeat their captured values. Every validation failure raises
        :class:`RuntimeError`. Returns the graph-owned output object(s)
        retained at capture time — the same identities on every call, with
        values updated in place.
        """
        if not self._captured:
            raise RuntimeError(
                "FastFP8CudaGraph has not captured a graph yet; call capture() "
                "before replay()"
            )
        _validate_input(self._args_spec, tuple(args), "args")
        _validate_input(self._kwargs_spec, kwargs, "kwargs")
        _copy_into_static(self._static_args, args)
        _copy_into_static(self._static_kwargs, kwargs)
        self._graph.replay()
        return self._static_out

    def __repr__(self) -> str:
        if self._captured:
            state = (
                f"captured on {self._device}, "
                f"workspace_bytes={self._workspace_bytes}"
            )
        else:
            state = "not captured"
        return f"{type(self).__name__}({state})"
