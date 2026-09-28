"""RaCo's fused residual block as a CUDA C++ kernel (``csrc/raco_resblock.cu``).

Implicit GEMM with ``mma.sync.m16n8k16`` (bf16 -> fp32): the A-fragments are ``ldmatrix``-ed straight from the NHWC
input / intermediate tile in shared memory (no im2col), the intermediate never leaves the SM, the 1x1 shortcut is a
tenth tap. 0.175 ms per 12-channel block at 1024x1024 vs 0.237 ms for the Triton kernel (``backends/triton/raco.py``, same
math and layout) and ~0.33 ms unfused; results agree with the Triton kernel to ~1e-5 relative.
"""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from flashim.backends.cuda import load

BLOCKS_PER_SM = 6


def available(device: torch.device | str = "cuda") -> bool:
    """True when the kernel can be built for ``device`` (needs ``nvcc`` and compute capability >= 8.0)."""
    if torch.device(device).type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        return False
    return _library() is not None


def _library() -> ctypes.CDLL | None:
    library = load("raco_resblock")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.launch_resblock.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        library.launch_resblock.restype = ctypes.c_int
        library._flashim_ready = True
    return library


def pack_block(triton_packed: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    """``(w1, b1, w2, b2, wm, bm)`` from :func:`flashim.ops._triton.raco.pack_block` / ``pad_bias`` ->
    ``(weights [19, 16, 16] bf16 (tap, cin, cout: 9 conv1, 9 conv2, shortcut), bias [32] fp32 (b1, b2 + bm))``."""
    w1, b1, w2, b2, wm, bm = triton_packed
    return torch.cat([w1, w2, wm[None]], 0).contiguous(), torch.cat([b1, b2 + bm]).float().contiguous()


def resblock(x: Tensor, cin: int, packed: tuple[Tensor, Tensor], scratch: Tensor | None = None) -> tuple[Tensor, None]:
    """``x``: NHWC bf16 ``[H, W, C_IN]`` (``C_IN`` in {4, 16}; padding channels must be zero). Returns
    ``(out [H, W, 16] bf16, None)`` — the second slot keeps the signature of the Triton kernel (which returns its scratch)."""
    h, w, in_ch = x.shape
    weights, bias = packed
    out = torch.empty(h, w, 16, device=x.device, dtype=torch.bfloat16)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    code = _library().launch_resblock(
        x.data_ptr(), out.data_ptr(), weights.data_ptr(), bias.data_ptr(), h, w, in_ch, sms * BLOCKS_PER_SM,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"raco_resblock launch failed (CUDA error {code}); in_ch must be 4 or 16, got {in_ch}")
    return out, None
