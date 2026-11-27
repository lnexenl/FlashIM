"""Triton kernels (NVIDIA; the in-tree kernels are validated on sm_80+ / sm_120)."""

import os

if manifest := os.environ.get("FLASHIM_RECORD_TRITON"):
    from ._record import install

    install(manifest)
