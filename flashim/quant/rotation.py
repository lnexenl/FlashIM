"""Orthogonal rotations that spread channel outliers before INT8 quantization (QuaRot-style): ``x W^T = (x R)(W R)^T``
for an orthogonal ``R``, so a rotation folded into the weights around a quantized input leaves the network's function
unchanged while the per-tensor scale sees evened-out channels."""

from __future__ import annotations

import torch
from torch import Tensor


def _hadamard_12(*, device: torch.device | str | None = None) -> Tensor:
    """Order-12 Paley Hadamard matrix (entries +-1, ``H @ H.T = 12 I``)."""
    residues = {(k * k) % 11 for k in range(1, 11)}
    h = torch.ones(12, 12, device=device, dtype=torch.float64)
    for i in range(11):
        for j in range(11):
            if i == j:
                h[i + 1, j + 1] = -1.0
            else:
                h[i + 1, j + 1] = 1.0 if (j - i) % 11 in residues else -1.0
    return h


def hadamard(n: int, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Normalized orthogonal Hadamard matrix ``[n, n]`` (``n = 2^k`` or ``12 * 2^k``, e.g. 96 for XFeat LighterGlue)."""
    if n < 1:
        raise ValueError(f"Hadamard order must be positive, got {n}")
    if n % 12 == 0 and (n // 12) & (n // 12 - 1) == 0:
        h = _hadamard_12(device=device)
    elif n & (n - 1) == 0:
        h = torch.ones(1, 1, device=device, dtype=torch.float64)
    else:
        raise ValueError(f"Hadamard needs 2^k or 12 * 2^k, got {n}")
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / n**0.5).to(dtype)
