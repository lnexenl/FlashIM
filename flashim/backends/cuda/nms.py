"""Fused CUDA C++ ``ops.simple_nms`` (``csrc/simple_nms.cu``): exact bit-identical two-round fast NMS in shared
memory using separable 1D max/OR (1 launch for ``radius=2``, 2 launches for ``radius=4``), replacing 16 full-res
DRAM passes (5x ``F.max_pool2d`` + masks) across SuperPoint, ALIKE, and ALIKED."""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from flashim import ops
from flashim.backends.cuda import load
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


def _library() -> ctypes.CDLL | None:
    library = load("simple_nms")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.launch_simple_nms.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        library.launch_simple_nms.restype = ctypes.c_int
        library._flashim_ready = True
    return library


@torch.compiler.assume_constant_result
def _available() -> bool:
    return _library() is not None


@torch.library.custom_op("flashim::simple_nms_cuda", mutates_args=())
def _simple_nms_cuda(scores: Tensor, radius: int) -> Tensor:
    sc = scores.contiguous()
    if sc.ndim == 3:
        b, h, w = sc.shape
    else:
        b, _, h, w = sc.shape
    out = torch.empty_like(sc)
    scratch = torch.empty((b, h, w), device=sc.device, dtype=torch.uint8) if radius == 4 else out
    code = _library().launch_simple_nms(
        sc.data_ptr(), scratch.data_ptr(), out.data_ptr(), b, h, w, radius,
        torch.cuda.current_stream(sc.device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"simple_nms launch failed (code {code}) for {tuple(sc.shape)} r={radius}")
    return out


@_simple_nms_cuda.register_fake
def _(scores: Tensor, radius: int) -> Tensor:
    return torch.empty_like(scores)


def _eligible(scores: Tensor, radius: int) -> bool:
    return (
        scores.is_cuda and scores.dtype == torch.float32
        and radius in (2, 4)
        and (scores.ndim == 3 or (scores.ndim == 4 and scores.shape[1] == 1))
        and scores.shape[-2] > 0 and scores.shape[-1] > 0
        and _available()
    )


@ops.simple_nms.register(
    "cuda", device=DeviceSpec("cuda"), requires=[Feature.NVCC], priority=1, when=_eligible
)
def _simple_nms_impl(scores: Tensor, radius: int) -> Tensor:
    return _simple_nms_cuda(scores, radius)
