"""Which ``libdevice`` makes Triton's ``erf`` (hence a fused GELU) reproduce torch's ``nn.GELU`` bit for bit.

Triton's bundled ``libdevice.10.bc`` and the one torch's CUDA build links differ in ``erff``: measured on an L20 and an
H20 (Triton 3.6, torch 2.11+cu130), 1.4% of the BF16 outputs of a Triton GELU differ from ``F.gelu`` (all at ``x < 0``,
where ``1 + erf`` cancels; ``erfc`` forms are worse, 6.9%) and CUDA 13.3's ``libdevice.10.bc`` still differs in 0.85%,
while CUDA 12.8's reproduces torch exactly. :func:`torch_gelu_libdevice` probes the candidates installed on the machine
against ``F.gelu`` once per process; kernels that must equal ``nn.GELU`` are launched with
``extern_libs={"libdevice": path}`` (:func:`gelu_launch_kwargs`) and are only eligible where the probe found a match.
"""

from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_PROBED: dict[int, str | None] = {}  # CUDA device index -> matching libdevice path, "" = the bundled one, None = none


@triton.jit
def gelu_erf(value):
    """``nn.GELU()`` (exact, erf) of an FP32 value: torch's ``x * 0.5 * (1 + erf(x * sqrt(1/2)))``."""
    return 0.5 * value * (1.0 + tl.math.erf(value * 0.7071067811865476))


@triton.jit
def _gelu_probe_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, gelu_erf(x).to(tl.bfloat16), mask=mask)


def _candidates() -> list[str | None]:
    roots = {os.environ.get(v) for v in ("CUDA_HOME", "CUDA_PATH")} - {None}
    paths = sorted(glob.glob("/usr/local/cuda*/nvvm/libdevice/libdevice.10.bc")
                   + glob.glob("/opt/cuda*/nvvm/libdevice/libdevice.10.bc")
                   + [os.path.join(r, "nvvm/libdevice/libdevice.10.bc") for r in roots])
    return [None] + [p for p in dict.fromkeys(paths) if os.path.exists(p)]  # None: Triton's own


def torch_gelu_libdevice(device: torch.device | str = "cuda") -> str | None:
    """Path of a ``libdevice.10.bc`` (``""``: Triton's bundled one) under which a Triton GELU equals ``F.gelu`` on
    BF16 inputs spanning the activation range, or ``None``. Cached per device; compiles one probe kernel per candidate."""
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    if index not in _PROBED:
        generator = torch.Generator(device=torch.device("cuda", index)).manual_seed(0)
        x = torch.cat([torch.randn(1 << 20, device=torch.device("cuda", index), generator=generator) * s
                       for s in (3.0, 0.5)]).bfloat16()
        reference = F.gelu(x)
        found = None
        for path in _candidates():
            y = torch.empty_like(x)
            kwargs = {} if path is None else {"extern_libs": {"libdevice": path}}
            _gelu_probe_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024, **kwargs)
            if torch.equal(y, reference):
                found = "" if path is None else path
                break
        _PROBED[index] = found
    return _PROBED[index]


def gelu_launch_kwargs(device: torch.device | str) -> dict:
    """Launch kwargs of a kernel that calls :func:`gelu_erf` and must equal ``nn.GELU`` (call only where
    :func:`torch_gelu_libdevice` is not ``None``)."""
    path = torch_gelu_libdevice(device)
    return {"extern_libs": {"libdevice": path}} if path else {}
