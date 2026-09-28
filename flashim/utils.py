"""Small module-surgery helpers shared by quantization and fusion passes."""

from __future__ import annotations

from torch import nn


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule at dotted path ``name`` (keeps sibling order and paths)."""
    parent, _, child = name.rpartition(".")
    (model.get_submodule(parent) if parent else model)._modules[child] = module


def nbytes(module: nn.Module) -> int:
    """Bytes held by a module's parameters and buffers."""
    return sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])
