"""Single-output-channel KxK convolution as a CUDA C++ kernel (``csrc/conv.cu``).

Register-blocked FP32 stencil: 8 x 4 outputs per thread from cp.async-staged input tiles widened in registers, all
channels' weights in shared memory. DISK's 80 -> 1 5x5 head at 1200x1600 (bf16 NCHW): 7.04 -> 1.80 ms on Thor
(sm_110), 1.42 -> 0.62 ms on an L20 (sm_89) vs the Triton stencil (``backends/triton/conv.py``), the same FP32
accumulation (max rel 2e-6 vs float64). 3x3: 3.10 -> 1.32 ms on Thor, a tie on the L20; 1x1 is left to Triton.
"""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from flashim import ops
from flashim.backends.cuda import load
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature

_DTYPES = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
MAX_CHANNELS = 256


def _library() -> ctypes.CDLL | None:
    library = load("conv")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.launch_conv_single_output.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_longlong] * 4 + [ctypes.c_void_p]
        )
        library.launch_conv_single_output.restype = ctypes.c_int
        library._flashim_ready = True
    return library


@torch.compiler.assume_constant_result  # a process constant: keeps dispatch traceable under torch.compile
def _available() -> bool:
    return _library() is not None


@torch.library.custom_op("flashim::conv2d_single_output_cuda", mutates_args=())
def conv2d_single_output(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """``x``: ``[B, C, H, W]`` with unit width stride and 16-byte multiples as the other strides; ``weight``: ``[1, C, k, k]`` ->
    ``[B, 1, H, W]`` FP32."""
    if x.data_ptr() % 16:  # a view at an unaligned offset (not traceable in ``when``): an aligned copy
        x = x.clone()
    batch, channels, height, width = x.shape
    out = torch.empty(batch, 1, height, width, device=x.device, dtype=torch.float32)
    w = weight.float().contiguous()
    b = bias.float().contiguous() if bias is not None else None
    code = _library().launch_conv_single_output(
        x.data_ptr(), w.data_ptr(), None if b is None else b.data_ptr(), out.data_ptr(), _DTYPES[x.dtype],
        weight.shape[-1], batch, channels, height, width, *x.stride(), torch.cuda.current_stream(x.device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"conv2d_single_output launch failed (code {code}) for {tuple(x.shape)} {x.stride()}")
    return out


@conv2d_single_output.register_fake
def _(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return x.new_empty(x.shape[0], 1, x.shape[2], x.shape[3], dtype=torch.float32)


def _eligible(x: Tensor, weight: Tensor, bias: Tensor | None) -> bool:
    if x.ndim != 4 or x.dtype not in _DTYPES or weight.shape[0] != 1 or weight.shape[1] != x.shape[1]:
        return False
    size = x.element_size()
    return (
        weight.shape[-1] == weight.shape[-2]
        and weight.shape[-1] in (3, 5)
        and x.shape[1] <= MAX_CHANNELS
        and x.stride(3) == 1
        and all(s * size % 16 == 0 for s in x.stride()[:3])
        and _available()  # nvcc build (once, cached); a failed build falls through to the next impl
    )


@ops.conv2d_single_output.register(
    "cuda", device=DeviceSpec("cuda", min_capability=(8, 0)),  # cp.async
    requires=[Feature.NVCC], priority=1, when=_eligible,
)
def _conv2d_single_output_impl(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return conv2d_single_output(x, weight, bias)
