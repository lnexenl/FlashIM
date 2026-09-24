"""Experimental: capture a fixed-shape forward into one CUDA graph and replay it.

Inputs are copied into static buffers captured with the graph; outputs are
graph-owned tensors that the next replay overwrites (clone them to keep them).
Shapes, dtypes and devices must match the capture exactly. Not enabled by any
preset: RoMaV2 never validated whole-model capture, so check outputs against eager
before relying on it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten


class CudaGraphRunner:
    def __init__(self, fn: Callable, *, warmups: int = 3) -> None:
        self.fn = fn
        self.warmups = warmups
        self.graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: list = []
        self._spec = None
        self._outputs = None

    @staticmethod
    def _signature(leaves: list) -> list:
        return [(tuple(x.shape), x.dtype, x.device) if isinstance(x, Tensor) else x for x in leaves]

    def capture(self, *args):
        leaves, spec = tree_flatten(args)
        if not any(isinstance(x, Tensor) and x.is_cuda for x in leaves):
            raise RuntimeError("CUDA graph capture needs CUDA tensor inputs")
        self._spec = spec
        self._static_inputs = [x.clone() if isinstance(x, Tensor) else x for x in leaves]
        self._signature_ref = self._signature(leaves)
        static_args = tree_unflatten(self._static_inputs, spec)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(self.warmups):
                self.fn(*static_args)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self._outputs = self.fn(*static_args)
        return self._outputs

    def __call__(self, *args):
        if self.graph is None:
            return self.capture(*args)
        leaves, spec = tree_flatten(args)
        if spec != self._spec or self._signature(leaves) != self._signature_ref:
            raise RuntimeError("CUDA graph replay inputs do not match the captured structure/shapes")
        with torch.inference_mode():
            for static, value in zip(self._static_inputs, leaves):
                if isinstance(static, Tensor):
                    static.copy_(value)
        self.graph.replay()
        return self._outputs
