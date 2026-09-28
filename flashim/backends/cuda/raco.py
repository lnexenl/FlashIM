"""RaCo's fused residual block as a CUDA C++ kernel (``csrc/raco_resblock.cu``).

Implicit GEMM with ``mma.sync.m16n8k16`` (bf16 -> fp32): the A-fragments are ``ldmatrix``-ed straight from the NHWC
input / intermediate tile in shared memory (no im2col), the intermediate never leaves the SM, the 1x1 shortcut is a
tenth tap. 0.175 ms per 12-channel block at 1024x1024 vs 0.237 ms for the Triton kernel (``backends/triton/raco.py``,
same math and layout) and ~0.33 ms unfused; results agree with the Triton kernel to ~1e-5 relative.
"""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from flashim import ops
from flashim.backends.cuda import load
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature

BLOCKS_PER_SM = 6
WIDTH = 16


def _library() -> ctypes.CDLL | None:
    library = load("raco_resblock")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.launch_resblock.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        library.launch_resblock.restype = ctypes.c_int
        library._flashim_ready = True
    return library


def _taps(conv: Tensor) -> Tensor:
    """``[C_out, C_in, k, k]`` -> ``[k*k, 16, 16]`` bf16 (tap, cin, cout), zero-padded."""
    out_ch, in_ch, k, _ = conv.shape
    out = torch.zeros(k * k, WIDTH, WIDTH, device=conv.device, dtype=torch.bfloat16)
    out[:, :in_ch, :out_ch] = conv.detach().permute(2, 3, 1, 0).reshape(k * k, in_ch, out_ch).to(torch.bfloat16)
    return out


def _bias(bias: Tensor) -> Tensor:
    out = torch.zeros(WIDTH, device=bias.device, dtype=torch.float32)
    out[: bias.numel()] = bias.detach().float()
    return out


def prepare(w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, wm: Tensor, bm: Tensor) -> tuple[Tensor, Tensor]:
    """Folded block parameters -> ``(weights [19, 16, 16] bf16 (tap, cin, cout: 9 conv1, 9 conv2, shortcut),
    bias [32] fp32 (b1, b2 + bm))``."""
    weights = torch.cat([_taps(w1), _taps(w2), _taps(wm)], 0).contiguous()
    return weights, torch.cat([_bias(b1), _bias(b2) + _bias(bm)]).contiguous()


@torch.library.custom_op("flashim::raco_resblock_cuda", mutates_args=())
def resblock(x: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    """``x``: NHWC bf16 ``[H, W, C_IN]`` (``C_IN`` in {4, 16}; padding channels must be zero) -> ``[H, W, 16]``."""
    h, w, in_ch = x.shape
    out = torch.empty(h, w, WIDTH, device=x.device, dtype=torch.bfloat16)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    code = _library().launch_resblock(
        x.data_ptr(), out.data_ptr(), weights.data_ptr(), bias.data_ptr(), h, w, in_ch, sms * BLOCKS_PER_SM,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"raco_resblock launch failed (CUDA error {code}); in_ch must be 4 or 16, got {in_ch}")
    return out


@resblock.register_fake
def _(x: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    return x.new_empty(x.shape[0], x.shape[1], WIDTH)


# -- registrations ---------------------------------------------------------------------------------------


def _eligible(x: Tensor, cin: int, w1: Tensor, *_) -> bool:
    return (
        x.ndim == 3
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[-1] in (4, 16)
        and w1.shape[0] <= WIDTH
        and _library() is not None  # nvcc build (once, cached); a failed build falls through to the next impl
    )


@ops.raco_resblock.register(
    "cuda", device=DeviceSpec("cuda"), requires=[Feature.NVCC, Feature.MMA_SYNC], priority=1, when=_eligible,
    prepare=prepare,
)
def _resblock_impl(x: Tensor, cin: int, weights: Tensor, bias: Tensor) -> Tensor:
    return resblock(x, weights, bias)
