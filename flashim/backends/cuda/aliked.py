"""Fused ALIKED ``score_head[2:]`` (``8->4->4->1`` 3x3 FP32 + SELU + sigmoid) as one CUDA C++ kernel
(``csrc/aliked_score_head.cu``). Eliminates two full-resolution ``[B, 4, H, W]`` DRAM round-trips and three slow
tiny-channel cuDNN kernels."""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from flashim import ops
from flashim.backends.cuda import load
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


def _library() -> ctypes.CDLL | None:
    library = load("aliked_score_head")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.launch_aliked_score_head.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        library.launch_aliked_score_head.restype = ctypes.c_int
        library._flashim_ready = True
    return library


@torch.compiler.assume_constant_result
def _available() -> bool:
    return _library() is not None


@torch.library.custom_op("flashim::aliked_score_head_cuda", mutates_args=())
def _aliked_score_head_cuda(head: Tensor, w1: Tensor, w2: Tensor, w3: Tensor, apply_sigmoid: bool) -> Tensor:
    head_c = head.float().contiguous()
    w1_c, w2_c, w3_c = w1.float().contiguous(), w2.float().contiguous(), w3.float().contiguous()
    b, _, h, w = head_c.shape
    out = torch.empty(b, 1, h, w, device=head_c.device, dtype=torch.float32)
    code = _library().launch_aliked_score_head(
        head_c.data_ptr(), w1_c.data_ptr(), w2_c.data_ptr(), w3_c.data_ptr(), out.data_ptr(),
        b, h, w, int(apply_sigmoid), torch.cuda.current_stream(head_c.device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"aliked_score_head launch failed (CUDA error {code}) for {tuple(head_c.shape)}")
    return out


@_aliked_score_head_cuda.register_fake
def _(head: Tensor, w1: Tensor, w2: Tensor, w3: Tensor, apply_sigmoid: bool) -> Tensor:
    return head.new_empty(head.shape[0], 1, head.shape[2], head.shape[3], dtype=torch.float32)


def _eligible(head: Tensor, w1: Tensor, w2: Tensor, w3: Tensor, *, apply_sigmoid: bool = True) -> bool:
    return (
        head.is_cuda and head.ndim == 4 and head.shape[1] == 8
        and head.dtype == torch.float32
        and tuple(w1.shape) == (4, 8, 3, 3)
        and tuple(w2.shape) == (4, 4, 3, 3)
        and tuple(w3.shape) == (1, 4, 3, 3)
        and _available()
    )


@ops.aliked_score_head.register(
    "cuda", device=DeviceSpec("cuda"), requires=[Feature.NVCC], priority=1, when=_eligible
)
def _aliked_score_head_impl(
    head: Tensor, w1: Tensor, w2: Tensor, w3: Tensor, *, apply_sigmoid: bool = True
) -> Tensor:
    return _aliked_score_head_cuda(head, w1, w2, w3, apply_sigmoid)
