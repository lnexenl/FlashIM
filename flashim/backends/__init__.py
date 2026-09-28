"""Kernel backends: optimized impls of :mod:`flashim.ops`, one package per kernel language or library.

``backends/<backend>/<category>.py`` mirrors ``ops/<category>.py``: it holds that backend's kernels *and* the
``ops.<op>.register(...)`` calls, with each impl's eligibility predicate next to it. ``ops/`` only declares ops
(signature, torch reference, docstring) and never imports a backend, so a new backend -- another GPU vendor, an
NPU, an out-of-tree package -- adds files here (or registers via the ``flashim.backends`` entry point) without
touching ``ops/``.

Rules:
  - registering must be cheap: frameworks with a heavy import (TileLang/TVM) keep their kernel builders in a
    private ``_*_kernels`` module that the impl imports on first call;
  - kernels opaque to ``torch.compile`` (CUDA C++, TileLang, vendor libraries) are wrapped in
    ``torch.library.custom_op`` with a fake; Triton kernels stay visible to Inductor;
  - hardware requirements go on the impl (``device=`` / ``requires=`` features), Python-package requirements on
    the :class:`~flashim.backends._loader.Backend` entry.

``cuda/`` (JIT-built CUDA C++) currently holds only RaCo's fused residual block, used directly by the model.
"""

from flashim.backends._loader import BACKENDS, LOADED, Backend, load

__all__ = ["BACKENDS", "LOADED", "Backend", "load"]
