"""Register-resident persistent CUDA C++ kernels for :data:`flashim.ops.log_optimal_transport` and
:data:`flashim.ops.sinkhorn_mutual` (``csrc/sinkhorn.cu``).

For ``M, N <= 1024`` on GPUs with enough resident CTAs, each warp owns one row and one column of
``scores * log2(e)`` in registers, broadcasting ``u`` / ``v`` (<= 4 KB) into shared memory once per CTA per
half-iteration and synchronizing across resident CTAs with a GPU-scope monotonic barrier (zero DRAM/L2 reads of the
score matrix, zero transpose workspace). Measured on RTX 5070 Ti (sm_120, iters=100): 0.275 ms at 256x256, 0.308 ms
at 512x512, 0.522 ms at 816x1024, 0.533 ms at 1024x1024 (25-58% faster than cooperative_groups + software exp2f +
transpose). Larger ``M, N > 1024`` fall back to the Triton persistent kernel for ``log_optimal_transport`` and the
L2-streaming cooperative kernel in the same library for ``sinkhorn_mutual``.
"""

from __future__ import annotations

import ctypes
import functools

import torch
from torch import Tensor

from flashim import ops
from flashim.backends.cuda import load
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature

_WARPS_PER_CTA = 8


def _library() -> ctypes.CDLL | None:
    library = load("sinkhorn")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.sinkhorn_max_resident_ctas.argtypes = [ctypes.c_int, ctypes.c_int]
        library.sinkhorn_max_resident_ctas.restype = ctypes.c_int
        library.sinkhorn_mutual_max_resident_ctas.argtypes = [ctypes.c_int, ctypes.c_int]
        library.sinkhorn_mutual_max_resident_ctas.restype = ctypes.c_int
        library.sinkhorn_cooperative_supported.argtypes = [ctypes.c_int]
        library.sinkhorn_cooperative_supported.restype = ctypes.c_int
        library.launch_sinkhorn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        library.launch_sinkhorn.restype = ctypes.c_int
        library.launch_sinkhorn_mutual_reg.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        library.launch_sinkhorn_mutual_reg.restype = ctypes.c_int
        library.launch_sinkhorn_mutual_stream.argtypes = (
            [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        )
        library.launch_sinkhorn_mutual_stream.restype = ctypes.c_int
        library._flashim_ready = True
    return library


@functools.cache
def _max_resident(device_index: int, max_c: int) -> int:
    lib = _library()
    if lib is None:
        return 0
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return int(lib.sinkhorn_max_resident_ctas(max_c, sm_count))


@functools.cache
def _max_resident_mutual(device_index: int, max_c: int) -> int:
    lib = _library()
    if lib is None:
        return 0
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return int(lib.sinkhorn_mutual_max_resident_ctas(max_c, sm_count))


@functools.cache
def _coop_supported(device_index: int) -> bool:
    lib = _library()
    return lib is not None and bool(lib.sinkhorn_cooperative_supported(device_index))


def _bucket(size: int) -> int:
    if size <= 256:
        return 256
    if size <= 512:
        return 512
    return 1024


# -- log_optimal_transport ---------------------------------------------------------------------------------


@torch.library.custom_op("flashim::log_optimal_transport_cuda", mutates_args=())
def _sinkhorn_cuda_op(scores: Tensor, alpha: Tensor, iters: int) -> Tensor:
    b, m, n = scores.shape
    device = scores.device
    max_mn = max(m, n)
    max_c = _bucket(max_mn)
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    max_res = _max_resident(dev_idx, max_c)
    num_ctas_dust = (max_mn + 1 + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA
    num_ctas = num_ctas_dust if b * num_ctas_dust <= max_res else (max_mn + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA
    alpha_t = alpha.to(device=device, dtype=torch.float32).reshape(()).contiguous()
    u = torch.empty((b, m + 1), device=device, dtype=torch.float32)
    v = torch.empty((b, n + 1), device=device, dtype=torch.float32)
    bar = torch.zeros((b,), device=device, dtype=torch.int32)
    out = torch.empty((b, m + 1, n + 1), device=device, dtype=torch.float32)
    code = _library().launch_sinkhorn(
        scores.data_ptr(),
        alpha_t.data_ptr(),
        u.data_ptr(),
        v.data_ptr(),
        bar.data_ptr(),
        out.data_ptr(),
        b,
        m,
        n,
        iters,
        num_ctas,
        max_c,
        torch.cuda.current_stream(device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"launch_sinkhorn failed with CUDA error {code}")
    return out


@_sinkhorn_cuda_op.register_fake
def _(scores: Tensor, alpha: Tensor, iters: int) -> Tensor:
    b, m, n = scores.shape
    return scores.new_empty((b, m + 1, n + 1))


def _eligible(scores: Tensor, alpha: Tensor, iters: int) -> bool:
    if not (
        scores.ndim == 3
        and scores.dtype == torch.float32
        and scores.is_contiguous()
        and iters >= 0
        and 0 < scores.shape[1] <= 1024
        and 0 < scores.shape[2] <= 1024
        and alpha.numel() == 1
    ):
        return False
    b, m, n = scores.shape
    dev_idx = scores.device.index if scores.device.index is not None else torch.cuda.current_device()
    num_ctas = (max(m, n) + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA
    max_res = _max_resident(dev_idx, _bucket(max(m, n)))
    return max_res > 0 and b * num_ctas <= max_res


@ops.log_optimal_transport.register(
    "cuda", device=DeviceSpec("cuda", min_capability=(8, 0)), requires=[Feature.NVCC], priority=1, when=_eligible
)
def _log_optimal_transport_cuda(scores: Tensor, alpha: Tensor, iters: int) -> Tensor:
    return _sinkhorn_cuda_op(scores, alpha, iters)


# -- sinkhorn_mutual ---------------------------------------------------------------------------------------


@torch.library.custom_op("flashim::sinkhorn_mutual_cuda", mutates_args=())
def _sinkhorn_mutual_cuda_op(sim: Tensor, bin_score: Tensor, iters: int = 100) -> tuple[Tensor, Tensor, Tensor]:
    """``sim [M, N]`` FP32 contiguous + scalar ``bin_score`` -> ``(prob [M], index [M], mutual [M])``."""
    m, n = sim.shape
    device = sim.device
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    bin_f32 = bin_score.to(device=device, dtype=torch.float32).reshape(()).contiguous()
    u = torch.empty(m + 1, device=device, dtype=torch.float32)
    v = torch.empty(n + 1, device=device, dtype=torch.float32)
    best_row = torch.empty(n, device=device, dtype=torch.int32)
    prob = torch.empty(m, device=device, dtype=torch.float32)
    index = torch.empty(m, device=device, dtype=torch.int64)
    mutual = torch.empty(m, device=device, dtype=torch.bool)
    stream = torch.cuda.current_stream(device).cuda_stream

    max_mn = max(m, n)
    if max_mn <= 1024:
        max_c = _bucket(max_mn)
        max_res = _max_resident_mutual(dev_idx, max_c)
        num_ctas_min = (max_mn + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA
        if max_res >= num_ctas_min:
            num_ctas_dust = (max_mn + 1 + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA
            num_ctas = num_ctas_dust if num_ctas_dust <= max_res else num_ctas_min
            bar = torch.zeros((1,), device=device, dtype=torch.int32)
            code = _library().launch_sinkhorn_mutual_reg(
                sim.data_ptr(),
                bin_f32.data_ptr(),
                u.data_ptr(),
                v.data_ptr(),
                bar.data_ptr(),
                best_row.data_ptr(),
                prob.data_ptr(),
                index.data_ptr(),
                mutual.data_ptr(),
                m,
                n,
                iters,
                num_ctas,
                max_c,
                stream,
            )
            if code != 0:
                raise RuntimeError(f"launch_sinkhorn_mutual_reg failed (CUDA error {code}) for {(m, n)} iters={iters}")
            return prob, index, mutual

    scores_t = torch.empty((n, m), device=device, dtype=torch.float32)
    code = _library().launch_sinkhorn_mutual_stream(
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
        stream,
    )
    if code != 0:
        raise RuntimeError(f"launch_sinkhorn_mutual_stream failed (CUDA error {code}) for {(m, n)} iters={iters}")
    return prob, index, mutual


@_sinkhorn_mutual_cuda_op.register_fake
def _(sim: Tensor, bin_score: Tensor, iters: int = 100) -> tuple[Tensor, Tensor, Tensor]:
    m = sim.shape[0]
    return (
        sim.new_empty(m, dtype=torch.float32),
        sim.new_empty(m, dtype=torch.int64),
        sim.new_empty(m, dtype=torch.bool),
    )


def _mutual_eligible(sim: Tensor, bin_score: Tensor, iters: int = 100) -> bool:
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
    max_mn = max(sim.shape[0], sim.shape[1])
    if max_mn <= 1024:
        max_res = _max_resident_mutual(dev_idx, _bucket(max_mn))
        if max_res >= (max_mn + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA:
            return True
    return _coop_supported(dev_idx)


@ops.sinkhorn_mutual.register(
    "cuda",
    device=DeviceSpec("cuda", min_capability=(8, 0)),
    requires=[Feature.NVCC],
    priority=1,
    when=_mutual_eligible,
)
def _sinkhorn_mutual_cuda(sim: Tensor, bin_score: Tensor, iters: int = 100) -> tuple[Tensor, Tensor, Tensor]:
    return _sinkhorn_mutual_cuda_op(sim, bin_score, iters)
