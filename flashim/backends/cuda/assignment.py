"""Register-resident persistent CUDA C++ kernel for :data:`flashim.ops.log_optimal_transport` (``csrc/sinkhorn.cu``).

Each warp owns one row and one column of ``scores * log2(e)`` in registers (``M, N <= 1024``), broadcasting the
tiny ``u`` / ``v`` dual vectors (<= 4 KB) into shared memory once per CTA per half-iteration and synchronizing across
resident CTAs with a GPU-scope monotonic barrier. Measured on RTX 5070 Ti (sm_120, iters=100): 0.266 ms at 256x256
(21.0x vs reference), 0.305 ms at 512x512 (18.3x), 0.508 ms at 816x1024 (11.1x), 0.512 ms at 1024x1024 (10.9x; 2.58x
faster than Triton multi-row, max log diff 4.8e-6, max prob diff 6.0e-7). Compile-checked on sm_80/89/90a/100a/120a
(validated on sm_120; other archs compiled, not validated). Falls back to the Triton persistent kernel when
``max(M, N) > 1024`` or when the GPU cannot keep ``B * cdiv(max(M, N), 8)`` CTAs resident simultaneously.
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
        library.launch_sinkhorn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        library.launch_sinkhorn.restype = ctypes.c_int
        library._flashim_ready = True
    return library


@functools.cache
def _max_resident(device_index: int, max_c: int) -> int:
    lib = _library()
    if lib is None:
        return 0
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return int(lib.sinkhorn_max_resident_ctas(max_c, sm_count))


def _bucket(size: int) -> int:
    if size <= 256:
        return 256
    if size <= 512:
        return 512
    return 1024


@torch.library.custom_op("flashim::log_optimal_transport_cuda", mutates_args=())
def _sinkhorn_cuda_op(scores: Tensor, alpha: Tensor, iters: int) -> Tensor:
    b, m, n = scores.shape
    device = scores.device
    max_mn = max(m, n)
    max_c = _bucket(max_mn)
    max_res = _max_resident(device.index or 0, max_c)
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
    num_ctas = (max(m, n) + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA
    max_res = _max_resident(scores.device.index or 0, _bucket(max(m, n)))
    return max_res > 0 and b * num_ctas <= max_res


@ops.log_optimal_transport.register(
    "cuda", device=DeviceSpec("cuda", min_capability=(8, 0)), requires=[Feature.NVCC], priority=1, when=_eligible
)
def _log_optimal_transport_cuda(scores: Tensor, alpha: Tensor, iters: int) -> Tensor:
    return _sinkhorn_cuda_op(scores, alpha, iters)
