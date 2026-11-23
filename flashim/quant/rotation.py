"""Orthogonal rotations that spread channel outliers before INT8 quantization (QuaRot-style): ``x W^T = (x R)(W R)^T``
for an orthogonal ``R``, so a rotation folded into the weights around a quantized input leaves the network's function
unchanged while the per-token or per-tensor scale sees evened-out channels.

``layernorm_orthogonal`` originates from ``model/mast3r`` (commit ``7741ccc``).
"""

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


def signed_hadamard(
    n: int,
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Orthogonal signed Sylvester Hadamard ``diag(s) @ H_n`` with deterministic Rademacher signs ``s in {-1, +1}^n``.

    Preserves ``R @ R^T = I`` and exact ``+-1/sqrt(n)`` BF16 entries (for even powers of two like ``n = 64``) while
    cancelling the all-ones DC row of ``H_n``, which otherwise creates a large DC spike on mostly-positive post-GELU
    activations.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).to(device=device, dtype=torch.float64)
    return (signs[:, None] * hadamard(n, device=device, dtype=torch.float64)).to(dtype)


def layernorm_orthogonal(
    n: int,
    *,
    block_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Symmetric orthogonal matrix ``Q [n, n]`` with ``Q @ 1 = 1`` so it commutes with un-affined ``LayerNorm``
    (``LN_0(x Q) = LN_0(x) Q``).

    Built as Kronecker products of Householder-like regular factors ``Q_k = (2/k) J_k - I_k`` (for ``n = 1024 = 4^5``
    every entry is ``+-1/32``; for ``n = 768 = 3 * 4^4`` max ``|Q_ij| = 1/24``). With ``block_size`` dividing ``n``,
    returns ``block_diag(Q_block, ...)`` (still satisfies ``Q @ 1 = 1``, keeping block-scaled quantization groups
    local).
    """
    if n < 1:
        raise ValueError(f"Expected positive dimension, got {n}")
    if block_size is not None:
        if block_size < 1 or n % block_size != 0:
            raise ValueError(f"block_size={block_size} must divide n={n}")
        qb = layernorm_orthogonal(block_size, device=device, dtype=torch.float64)
        return torch.block_diag(*([qb] * (n // block_size))).to(dtype)
    factors: list[int] = []
    rem = n
    while rem % 4 == 0:
        factors.append(4)
        rem //= 4
    if rem > 1:
        factors.append(rem)
    q = torch.ones((1, 1), device=device, dtype=torch.float64)
    for k in factors:
        qk = (2.0 / k) * torch.ones((k, k), device=device, dtype=torch.float64) - torch.eye(
            k, device=device, dtype=torch.float64
        )
        q = torch.kron(q, qk)
    return q.to(dtype)
