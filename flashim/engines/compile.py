"""Region compilation: ``torch.compile`` selected submodules' bound ``forward``.

Compiling bound methods (instead of wrapping modules) keeps module paths and
``state_dict`` keys unchanged. ``fullgraph=False`` lets custom ops graph-break.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

_MARKER = "_flashim_compiled"


def compile_regions(
    model: nn.Module,
    regions: Iterable[str],
    *,
    mode: str | None = "max-autotune-no-cudagraphs",
    dynamic: bool = False,
) -> list[str]:
    if model.training:
        raise RuntimeError("compile_regions requires model.eval()")
    import torch._inductor.config as inductor_config

    inductor_config.triton.cudagraphs = False
    compiled = []
    for name in regions:
        region = model.get_submodule(name)
        if region.__dict__.get(_MARKER):
            continue
        region.forward = torch.compile(region.forward, fullgraph=False, dynamic=dynamic, mode=mode)
        region.__dict__[_MARKER] = True
        compiled.append(name)
    return compiled
