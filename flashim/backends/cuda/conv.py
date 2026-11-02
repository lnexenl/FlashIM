"""Convolution kernels in CUDA C++: the single-output-channel stencil (``csrc/conv.cu``) and the depthwise 5x5 blocks
(``csrc/depthwise.cu``, the end of this file).

Single-output-channel KxK convolution (``csrc/conv.cu``).

Register-blocked FP32 stencil: 8 x 4 outputs per thread from cp.async-staged input tiles widened in registers, all
channels' weights in shared memory. DISK's 80 -> 1 5x5 head at 1200x1600 (bf16 NCHW): 7.04 -> 1.80 ms on Thor
(sm_110), 1.42 -> 0.62 ms on an L20 (sm_89) vs the Triton stencil (``backends/triton/conv.py``), the same FP32
accumulation (max rel 2e-6 vs float64). 3x3: 3.10 -> 1.32 ms on Thor, a tie on the L20; 1x1 is left to Triton.

Depthwise 5x5 (``csrc/depthwise.cu``), bit-identical to the Triton kernels (``torch.equal`` on every tile, DeDoDe's real
shapes and ragged ones; ``tests/ops/test_depthwise_cuda.py``), registered for sm_89 / sm_90 only (the architectures
measured; compiled, not run, for sm_80 / 86 / 87 / 100 / 120):

- ``depthwise5x5_relu_int8``, the INT8 producer of a packed 1x1 (any ``C % 64 == 0``), in situ per call (CUDA-graph
  replay, DeDoDe at 1200 px, 512 x 112 x 150): L20 0.053 ms (Triton) -> 0.021, H20 0.083 (TileLang) -> 0.031; it also
  covers the 256 / 768-channel blocks that had no kernel on the L20.
- ``depthwise5x5_relu_pointwise``, the fused BF16 block (``C`` in 32 / 64 / 128): ``mma.sync`` m16n8k16 started from the
  bias, as Triton lowers ``dot + bias``. L20 6.8 -> 6.1 ms and H20 9.5 -> 5.6 ms per DeDoDe extract (the L20 is
  DRAM-bound on the full-resolution blocks).
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


# -- depthwise 5x5 kernels (``csrc/depthwise.cu``) --------------------------------------------------------------

#: ``(COLS, TH, CY)`` of the INT8 producer per capability, from in-situ tile sweeps (CUDA-graph replay, L2 warm) on
#: DeDoDe's real shapes at 1200 px (512 x 112 x 150, 256 x 224 x 300, 768 x 64 x 84, and the odd-width dedode-g ones),
#: 54 tiles: all bit-identical to Triton; (4, 8) won on both GPUs, ``CY`` (column groups per CTA) 2 on the L20 and 1 on
#: the H20, within 1% of CY 4 (the tile LoMa's prototype used).
PRODUCER_TILE = {(8, 9): (4, 8, 2), (9, 0): (4, 8, 1)}
#: ``(COLS, TH, NG)`` of the fused BF16 block per capability and channel count (same sweeps, 5 tiles x 2 occupancy caps,
#: all bit-identical to Triton). The L20 is DRAM-bound on the full-resolution 32 / 64-channel blocks (every tile within
#: 3% of the best, ~10% over Triton there); one tile for every channel count is within 0.3% of the per-shape best.
DWPW_TILE = {
    (8, 9): {32: (4, 4, 4), 64: (4, 4, 4), 128: (4, 4, 4)},
    (9, 0): {32: (4, 8, 4), 64: (4, 8, 4), 128: (4, 8, 8)},
}
_DWPW_CHANNELS = (32, 64, 128)
_GRID_LIMIT = 65535


def _depthwise_library() -> ctypes.CDLL | None:
    library = load("depthwise")
    if library is not None and not getattr(library, "_flashim_ready", False):
        library.launch_dw5x5_relu_int8.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        library.launch_dw5x5_relu_int8.restype = ctypes.c_int
        library.launch_dwpw.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        library.launch_dwpw.restype = ctypes.c_int
        library._flashim_ready = True
    return library


@torch.compiler.assume_constant_result
def _depthwise_available() -> bool:
    return _depthwise_library() is not None


def launch_depthwise5x5_relu_int8(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor, tile: tuple[int, int, int]) -> Tensor:
    """The producer kernel with an explicit ``(COLS, TH, CY)`` tile (sweeps and tests; dispatch uses ``PRODUCER_TILE``)."""
    batch, channels, height, width = value.shape
    out = torch.empty((batch * height * width, channels), device=value.device, dtype=torch.int8)
    code = _depthwise_library().launch_dw5x5_relu_int8(
        value.data_ptr(), weight.data_ptr(), bias.data_ptr(), scale.data_ptr(), out.data_ptr(), batch, height, width,
        channels, *tile, torch.cuda.current_stream(value.device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"depthwise5x5_relu_int8 launch failed (code {code}) for {tuple(value.shape)} tile {tile}")
    return out


@torch.library.custom_op("flashim::depthwise5x5_relu_int8_cuda", mutates_args=())
def _depthwise5x5_relu_int8(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    if value.data_ptr() % 4:  # bf16x2 loads: a view at an odd element offset (not traceable in ``when``)
        value = value.clone(memory_format=torch.channels_last)
    if weight.data_ptr() % 4:
        weight = weight.clone()
    tile = PRODUCER_TILE[torch.cuda.get_device_capability(value.device)]
    return launch_depthwise5x5_relu_int8(value, weight, bias, scale.reshape(1), tile)


@_depthwise5x5_relu_int8.register_fake
def _(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    batch, channels, height, width = value.shape
    return value.new_empty((batch * height * width, channels), dtype=torch.int8)


def _producer_eligible(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> bool:
    if value.ndim != 4 or value.dtype != torch.bfloat16 or value.shape[1] % 64:
        return False
    batch, channels, height, _ = value.shape
    return (
        value.is_contiguous(memory_format=torch.channels_last)
        and tuple(weight.shape) == (25, channels) and weight.dtype == torch.bfloat16 and weight.is_contiguous()
        and tuple(bias.shape) == (channels,) and bias.dtype == torch.float32 and bias.is_contiguous()
        and scale.dtype == torch.float32 and scale.numel() == 1
        and batch * -(-height // 4) <= _GRID_LIMIT
        and _depthwise_available()  # nvcc build (once, cached); a failed build falls through to the next impl
    )


def _dwpw_eligible(value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor) -> bool:
    if value.ndim != 4 or value.dtype != torch.bfloat16 or value.shape[1] not in _DWPW_CHANNELS:
        return False
    batch, channels, height, _ = value.shape
    return (
        value.is_contiguous(memory_format=torch.channels_last)
        and tuple(dw_weight.shape) == (25, channels) and dw_weight.dtype == torch.bfloat16 and dw_weight.is_contiguous()
        and tuple(dw_bias.shape) == (channels,) and dw_bias.dtype == torch.bfloat16 and dw_bias.is_contiguous()
        and tuple(pw_weight.shape) == (channels, channels) and pw_weight.dtype == torch.bfloat16
        and pw_weight.is_contiguous()
        and tuple(pw_bias.shape) == (channels,) and pw_bias.dtype == torch.bfloat16 and pw_bias.is_contiguous()
        and batch * -(-height // 4) <= _GRID_LIMIT
        and _depthwise_available()
    )


def launch_depthwise5x5_relu_pointwise(
    value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor, tile: tuple[int, int, int]
) -> Tensor:
    """The fused block kernel with an explicit ``(COLS, TH, NG)`` tile: BF16 ``[B, H, W, C]``."""
    batch, channels, height, width = value.shape
    out = torch.empty((batch, height, width, channels), device=value.device, dtype=torch.bfloat16)
    cols, th, ng = tile
    code = _depthwise_library().launch_dwpw(
        value.data_ptr(), dw_weight.data_ptr(), dw_bias.data_ptr(), pw_weight.data_ptr(), pw_bias.data_ptr(),
        out.data_ptr(), batch, height, width, channels, cols, th, ng,
        torch.cuda.current_stream(value.device).cuda_stream,
    )
    if code != 0:
        raise RuntimeError(f"depthwise5x5_relu_pointwise launch failed (code {code}) for {tuple(value.shape)} tile {tile}")
    return out


@torch.library.custom_op("flashim::depthwise5x5_relu_pointwise_cuda", mutates_args=())
def _depthwise5x5_relu_pointwise(
    value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor
) -> Tensor:
    if value.data_ptr() % 4:
        value = value.clone(memory_format=torch.channels_last)
    if dw_weight.data_ptr() % 4 or pw_weight.data_ptr() % 4:
        dw_weight, pw_weight = dw_weight.clone(), pw_weight.clone()
    return launch_depthwise5x5_relu_pointwise(value, dw_weight, dw_bias, pw_weight, pw_bias, DWPW_TILE[torch.cuda.get_device_capability(value.device)][value.shape[1]])


@_depthwise5x5_relu_pointwise.register_fake
def _(value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor) -> Tensor:
    batch, channels, height, width = value.shape
    return value.new_empty((batch, height, width, channels), dtype=torch.bfloat16)


# sm_89 (L20) and sm_90 (H20): the architectures this was measured on. Compiled for sm_80 / 86 / 87 / 100 / 120 but not
# run there, so those keep their Triton / TileLang impls.
_MEASURED = DeviceSpec("cuda", min_capability=(8, 9), max_capability=(9, 0))


@ops.depthwise5x5_relu_int8.register("cuda", device=_MEASURED, requires=[Feature.NVCC], priority=2, when=_producer_eligible)
def _depthwise5x5_relu_int8_impl(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    return _depthwise5x5_relu_int8(value, weight, bias, scale)


@ops.depthwise5x5_relu_pointwise.register(
    "cuda", device=_MEASURED, requires=[Feature.NVCC, Feature.MMA_SYNC], priority=1, when=_dwpw_eligible
)
def _depthwise5x5_relu_pointwise_impl(
    value: Tensor, dw_weight: Tensor, dw_bias: Tensor, pw_weight: Tensor, pw_bias: Tensor
) -> Tensor:
    return _depthwise5x5_relu_pointwise(value, dw_weight, dw_bias, pw_weight, pw_bias).permute(0, 3, 1, 2)
