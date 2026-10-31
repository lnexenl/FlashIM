"""Orthogonal rotations that spread channel outliers before INT8 quantization (QuaRot-style): ``x W^T = (x R)(W R)^T``
for an orthogonal ``R``, so a rotation folded into the weights around a quantized input leaves the network's function
unchanged while the per-tensor scale sees evened-out channels."""

from __future__ import annotations

import torch
from torch import Tensor


def hadamard(n: int, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """The normalized Sylvester Hadamard matrix ``[n, n]`` (``n`` a power of two): symmetric and orthogonal."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Sylvester Hadamard needs a power of two, got {n}")
    h = torch.ones(1, 1, device=device, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / n**0.5).to(dtype)
