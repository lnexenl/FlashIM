"""Single-launch cooperative Log-Sinkhorn optimal transport + mutual argmax (``csrc/sinkhorn.cu``).

Replaces ~200+ eager PyTorch kernel launches and intermediate ``[M+1, N+1]`` allocations with one
cooperative CUDA kernel launch. For ``M, N <= 1024`` on GPUs with enough active warps, each warp keeps
its row and column of ``sim`` in registers across all iterations (``sim`` is read from global memory once)
and stages ``u`` / ``v`` in shared memory. Larger shapes or smaller GPUs run the L2-streaming cooperative
fallback in the same kernel library.
"""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from flashim import ops
from flashim.backends.cuda import load
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


def _library() -> ctypes.CDLL | None:
    library = load("sinkhorn")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.sinkhorn_cooperative_supported.argtypes = [ctypes.c_int]
        library.sinkhorn_cooperative_supported.restype = ctypes.c_int
        library.launch_sinkhorn_mutual.argtypes = (
            [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        )
        library.launch_sinkhorn_mutual.restype = ctypes.c_int
        library._flashim_ready = True
    return library


@torch.compiler.assume_constant_result
def _available(device_index: int) -> bool:
    lib = _library()
    return lib is not None and bool(lib.sinkhorn_cooperative_supported(device_index))


@torch.library.custom_op("flashim::sinkhorn_mutual_cuda", mutates_args=())
def sinkhorn_mutual(sim: Tensor, bin_score: Tensor, iters: int = 100) -> tuple[Tensor, Tensor, Tensor]:
    """``sim [M, N]`` FP32 contiguous + scalar ``bin_score`` -> ``(prob [M], index [M], mutual [M])``."""
    m, n = sim.shape
    device = sim.device
    bin_f32 = bin_score.float().reshape(())
    if not bin_f32.is_contiguous():
        bin_f32 = bin_f32.contiguous()
    scores_t = torch.empty((n, m), device=device, dtype=torch.float32)
    u = torch.empty(m + 1, device=device, dtype=torch.float32)
    v = torch.empty(n + 1, device=device, dtype=torch.float32)
    best_row = torch.empty(n, device=device, dtype=torch.int32)
    prob = torch.empty(m, device=device, dtype=torch.float32)
    index = torch.empty(m, device=device, dtype=torch.int64)
    mutual = torch.empty(m, device=device, dtype=torch.bool)
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    code = _library().launch_sinkhorn_mutual(
        sim.data_ptr(),
        bin_f32.data_ptr(),
        scores_t.data_ptr(),
        u.data_ptr(),
        v.data_ptr(),
        best_row.data_ptr(),
        prob.data_ptr(),
        index.data_ptr(),
        mutual.data_ptr(),
        m,
        n,
        iters,
        dev_idx,
        torch.cuda.current_stream(device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"sinkhorn_mutual launch failed (CUDA error {code}) for shape {(m, n)} iters={iters}")
    return prob, index, mutual


@sinkhorn_mutual.register_fake
def _(sim: Tensor, bin_score: Tensor, iters: int = 100) -> tuple[Tensor, Tensor, Tensor]:
    m = sim.shape[0]
    return (
        sim.new_empty(m, dtype=torch.float32),
        sim.new_empty(m, dtype=torch.int64),
        sim.new_empty(m, dtype=torch.bool),
    )


def _eligible(sim: Tensor, bin_score: Tensor, iters: int = 100) -> bool:
    if (
        sim.ndim != 2
        or sim.dtype != torch.float32
        or not sim.is_contiguous()
        or sim.numel() == 0
        or bin_score.device != sim.device
        or bin_score.numel() != 1
        or iters < 0
    ):
        return False
    dev_idx = sim.device.index if sim.device.index is not None else torch.cuda.current_device()
    return _available(dev_idx)


@ops.sinkhorn_mutual.register(
    "cuda",
    device=DeviceSpec("cuda", min_capability=(8, 0)),
    requires=[Feature.NVCC],
    priority=1,
    when=_eligible,
)
def _sinkhorn_mutual_impl(sim: Tensor, bin_score: Tensor, iters: int = 100) -> tuple[Tensor, Tensor, Tensor]:
    return sinkhorn_mutual(sim, bin_score, iters)
