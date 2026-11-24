"""Exact 3D Euclidean reciprocal nearest neighbors via CUDA C++ complete binary AABB tree (``csrc/reciprocal_nn.cu``).

Sorts each 3D point cloud along a 30-bit Morton curve (after monotone ``signed_log1p`` compression for exponential
depth ranges), packs ``(x, y, z, orig_idx)`` into 16-byte aligned ``PtIdx`` records, builds a complete binary AABB
tree over 16-point leaves, and runs near-child-first per-thread traversal with a distance-tagged stack and
``__ldg`` vectorized loads (zero host sync). Exact (bit-identical to brute-force L2 with smallest-original-index
tie breaking).

Measured on sm_120 (RTX 5070 Ti, DUSt3R 3D pointmaps): Toronto 196k x 196k 27.63 ms (Triton) -> 1.92 ms (14.4x);
Sacre 180k x 196k 9.49 ms -> 2.35 ms (4.0x); ``LEAF=16`` beat ``LEAF=8`` (1.86 / 2.79 ms) and ``LEAF=32`` (2.22 /
2.84 ms). ``ncu`` SpeedOfLight on ``nn_bvh_query_kernel``: L1/TEX 68.2%, DRAM 2.65%, 1.15 ms/direction.
"""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from flashim import ops
from flashim.backends.cuda import load
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature

LEAF = 16


def _library() -> ctypes.CDLL | None:
    library = load("reciprocal_nn")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.launch_morton3d.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
        library.launch_morton3d.restype = ctypes.c_int
        library.launch_build_bvh.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        library.launch_build_bvh.restype = ctypes.c_int
        library.launch_query_bvh.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        library.launch_query_bvh.restype = ctypes.c_int
        library._flashim_ready = True
    return library


@torch.compiler.assume_constant_result
def _available() -> bool:
    return _library() is not None


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_bvh(pts: Tensor, bounds: Tensor) -> tuple[Tensor, Tensor, int, int]:
    lib = _library()
    n = pts.shape[0]
    device = pts.device
    stream = torch.cuda.current_stream(device).cuda_stream
    pts_c = pts.contiguous()
    codes = torch.empty(n, dtype=torch.int32, device=device)
    code = lib.launch_morton3d(pts_c.data_ptr(), bounds.data_ptr(), codes.data_ptr(), n, stream)
    if code != 0:
        raise RuntimeError(f"launch_morton3d failed (CUDA error {code})")
    order = torch.argsort(codes)
    n_leaves = (n + LEAF - 1) // LEAF
    p_size = _next_pow2(n_leaves)
    # PtIdx = 16 bytes (4 int32s), AABB = 32 bytes (8 float32s)
    packed = torch.empty((n, 4), dtype=torch.int32, device=device)
    nodes = torch.empty((2 * p_size, 8), dtype=torch.float32, device=device)
    code = lib.launch_build_bvh(
        pts_c.data_ptr(), order.data_ptr(), packed.data_ptr(), nodes.data_ptr(), n, n_leaves, p_size, stream
    )
    if code != 0:
        raise RuntimeError(f"launch_build_bvh failed (CUDA error {code})")
    return packed, nodes, n_leaves, p_size


@torch.library.custom_op("flashim::reciprocal_nn_3d_cuda", mutates_args=())
def _reciprocal_nn_3d_cuda(pts1: Tensor, pts2: Tensor) -> tuple[Tensor, Tensor]:
    device = pts1.device
    n1, n2 = pts1.shape[0], pts2.shape[0]
    if n1 == 0 or n2 == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty

    lo = torch.minimum(pts1.amin(dim=0), pts2.amin(dim=0))
    hi = torch.maximum(pts1.amax(dim=0), pts2.amax(dim=0))
    lo_log = lo.sign() * torch.log1p(lo.abs())
    hi_log = hi.sign() * torch.log1p(hi.abs())
    inv_span = 1023.0 / (hi_log - lo_log).clamp(min=1e-6)
    bounds = torch.cat((lo_log, inv_span)).contiguous()

    packed1, nodes1, leaves1, p1 = _build_bvh(pts1, bounds)
    packed2, nodes2, leaves2, p2 = _build_bvh(pts2, bounds)

    lib = _library()
    stream = torch.cuda.current_stream(device).cuda_stream
    nn1_in_2 = torch.empty(n1, dtype=torch.int64, device=device)
    nn2_in_1 = torch.empty(n2, dtype=torch.int64, device=device)
    code = lib.launch_query_bvh(
        packed1.data_ptr(), packed2.data_ptr(), nodes2.data_ptr(), nn1_in_2.data_ptr(),
        n1, n2, leaves2, p2, stream,
    )
    if code != 0:
        raise RuntimeError(f"launch_query_bvh (1->2) failed (CUDA error {code})")
    code = lib.launch_query_bvh(
        packed2.data_ptr(), packed1.data_ptr(), nodes1.data_ptr(), nn2_in_1.data_ptr(),
        n2, n1, leaves1, p1, stream,
    )
    if code != 0:
        raise RuntimeError(f"launch_query_bvh (2->1) failed (CUDA error {code})")

    idx2 = torch.arange(n2, device=device, dtype=torch.int64)
    reciprocal = nn1_in_2[nn2_in_1] == idx2
    return nn2_in_1[reciprocal], idx2[reciprocal]


@_reciprocal_nn_3d_cuda.register_fake
def _(pts1: Tensor, pts2: Tensor) -> tuple[Tensor, Tensor]:
    ctx = torch.library.get_ctx()
    m = ctx.new_dynamic_size()
    return pts1.new_empty((m,), dtype=torch.int64), pts2.new_empty((m,), dtype=torch.int64)


def _eligible(pts1: Tensor, pts2: Tensor, block_size: int = 4096) -> bool:
    del block_size
    return (
        pts1.ndim == 2
        and pts2.ndim == 2
        and pts1.shape[1] == 3
        and pts2.shape[1] == 3
        and pts1.dtype == torch.float32
        and pts2.dtype == torch.float32
        and 0 < pts1.shape[0] < 2**31
        and 0 < pts2.shape[0] < 2**31
        and _available()
    )


@ops.reciprocal_nn.register(
    "cuda", device=DeviceSpec("cuda"), requires=[Feature.NVCC], priority=1, when=_eligible
)
def _reciprocal_nn_cuda_impl(pts1: Tensor, pts2: Tensor, block_size: int = 4096) -> tuple[Tensor, Tensor]:
    del block_size
    return _reciprocal_nn_3d_cuda(pts1, pts2)
