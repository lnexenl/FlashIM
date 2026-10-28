"""Small module-surgery helpers shared by quantization and fusion passes, cached device constants, and a CPU-safe
``grid_sample``."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule at dotted path ``name`` (keeps sibling order and paths)."""
    parent, _, child = name.rpartition(".")
    (model.get_submodule(parent) if parent else model)._modules[child] = module


def nbytes(module: nn.Module) -> int:
    """Bytes held by a module's parameters and buffers."""
    return sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])


_CONSTANTS: dict[tuple, Tensor] = {}
_MAX_CONSTANTS = 1024


def device_constant(values: Sequence[float], like: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """``like.new_tensor(values, dtype=dtype)``, created once per (device, dtype, values) and reused.

    ``new_tensor`` on a CUDA tensor is a blocking host -> device copy, which synchronizes the stream: per call it keeps
    the CPU from queueing the next stage while the GPU works (``benchmarks/profile/sync_audit.py`` lists them). The
    cached tensor is created outside inference mode so it can be used in and out of it; callers must not modify it.
    """
    dtype = like.dtype if dtype is None else dtype
    key = (like.device, dtype, tuple(values))
    constant = _CONSTANTS.get(key)
    if constant is None:
        if len(_CONSTANTS) >= _MAX_CONSTANTS:  # image-size dependent values: bound the cache
            _CONSTANTS.clear()
        with torch.inference_mode(False):
            constant = _CONSTANTS[key] = torch.tensor(list(values), dtype=dtype, device=like.device)
    return constant


def grid_sample(input: Tensor, grid: Tensor, **kwargs) -> Tensor:
    """``F.grid_sample``, computed in FP32 for BF16 / FP16 inputs on the CPU: torch 2.11's reduced-precision CPU kernel
    segfaults on large inputs (measured: any ``[1, C, 704, 1024]`` BF16 or FP16 input, C = 16 or 128, crashes within a
    few calls; FP32 does not). Other devices and dtypes call it unchanged."""
    if input.device.type == "cpu" and input.dtype in (torch.bfloat16, torch.float16):
        return F.grid_sample(input.float(), grid.float(), **kwargs).to(input.dtype)
    return F.grid_sample(input, grid, **kwargs)
