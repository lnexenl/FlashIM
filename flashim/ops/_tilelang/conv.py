"""TileLang depthwise 5x5 kernels (sm_90+: TMA + mbarrier pipeline).

``depthwise5x5_relu_mxfp8``: same math and summation order as the Triton kernel
(bit-identical output), but input rows are staged in shared memory by a warp-specialized
TMA pipeline instead of being re-read from L1, and each thread keeps its 25 weights in
registers. One thread owns one column and two channels; per output row, 16 lanes reduce
the 32-channel MX group amax with shuffles.
"""

# No ``from __future__ import annotations``: TileLang evaluates the T.Tensor annotations
# of a prim_func in its defining scope.
import functools

import tilelang
import tilelang.language as T
import torch
from torch import Tensor

# Plain (generic-proxy) shared-memory reads must be ordered before the TMA (async-proxy)
# write that later refills the stage. TileLang 0.1.14 arrives on the stage's "empty"
# barrier right after the last buffer access, without this fence, and a warp's LDS can
# still be in flight when the producer's TMA overwrites the stage (seen as rare, corrupt
# rows). Taking the buffer pointer makes TileLang treat the call as that last access.
_RELEASE_READS_SRC = """
__device__ __forceinline__ void flashim_release_smem_reads(const void*) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
"""

COLS = 16  # output columns per block
GROUPS_PER_BLOCK = 2  # 32-channel MX groups per block
STAGES = 4


def _tile_h(height: int) -> int:
    """Rows per block: the fewest strips of at most 256 rows, split evenly."""
    strips = -(-height // 256)
    return -(-height // strips)


@tilelang.jit
def _build_depthwise5x5_relu_mxfp8(batch: int, height: int, width: int, channels: int, tile_h: int):
    groups = channels // 32
    scale_tiles = (groups + 3) // 4
    rows = batch * height * width
    padded_rows = (rows + 127) // 128 * 128
    block_groups = groups // GROUPS_PER_BLOCK

    @T.prim_func
    def main(
        x: T.Tensor((batch, height, width, channels), T.bfloat16),
        weight: T.Tensor((25 * channels,), T.bfloat16),
        bias: T.Tensor((channels,), T.float32),
        out: T.Tensor((rows * channels,), T.float8_e4m3fn),
        scale: T.Tensor((padded_rows * scale_tiles * 4,), T.uint8),
    ):
        with T.Kernel(
            T.ceildiv(width, COLS), T.ceildiv(height, tile_h), batch * block_groups,
            threads=16 * COLS * GROUPS_PER_BLOCK,
        ) as (bx, by, bz):
            T.no_set_max_nreg()
            T.import_source(_RELEASE_READS_SRC)
            tx = T.get_thread_binding()
            lane = tx % 16
            tcol = tx // (16 * GROUPS_PER_BLOCK)
            gi = (tx // 16) % GROUPS_PER_BLOCK
            col = bx * COLS + tcol
            n = bz // block_groups
            g0 = (bz % block_groups) * GROUPS_PER_BLOCK
            g = g0 + gi
            c0 = g * 32 + lane * 2
            w = T.alloc_local((50,), T.float32)
            acc = T.alloc_local((10,), T.float32)
            raw = T.alloc_local((10,), T.bfloat16)
            b = T.alloc_local((2,), T.float32)
            q = T.alloc_local((2,), T.float8_e4m3fn)
            m = T.alloc_var(T.float32)
            e = T.alloc_var(T.float32)
            row_s = T.alloc_shared((COLS + 4, 32 * GROUPS_PER_BLOCK), T.bfloat16)
            for k in T.unroll(25):
                for i in T.unroll(2):
                    w[k * 2 + i] = T.cast(weight[k * channels + c0 + i], T.float32)
            for i in T.unroll(2):
                b[i] = bias[c0 + i]
            for i in T.unroll(10):
                acc[i] = T.float32(0)
            for step in T.Pipelined(tile_h + 4, num_stages=STAGES):
                # TMA zero-fills the out-of-image halo (conv padding 2).
                T.copy(
                    x[n, by * tile_h - 2 + step, bx * COLS - 2 : bx * COLS + COLS + 2,
                      g0 * 32 : g0 * 32 + 32 * GROUPS_PER_BLOCK],
                    row_s,
                )
                for kx in T.unroll(5):
                    for i in T.vectorized(2):
                        raw[kx * 2 + i] = row_s[tcol + kx, gi * 32 + lane * 2 + i]
                T.call_extern("handle", "flashim_release_smem_reads", T.access_ptr(row_s[0, 0], "r"))
                # Row rolling: input row iy feeds output row iy + 2 - ky through tap ky
                # (slot 4 - ky); per output the FP32 order is ky-major, kx-minor.
                for kx in T.unroll(5):
                    for ky in T.unroll(5):
                        for i in T.unroll(2):
                            acc[(4 - ky) * 2 + i] = (
                                acc[(4 - ky) * 2 + i] + T.cast(raw[kx * 2 + i], T.float32) * w[(ky * 5 + kx) * 2 + i]
                            )
                out_y = by * tile_h - 4 + step
                if step >= 4 and out_y < height:
                    a0 = T.max(acc[0] + b[0], T.float32(0))
                    a1 = T.max(acc[1] + b[1], T.float32(0))
                    m = T.max(a0, a1)  # ReLU output: non-negative
                    for delta in T.unroll(4):
                        m = T.max(m, T.shfl_xor(m, 8 >> delta))
                    # Same ops as the Triton kernel: approximate division, libdevice log2f.
                    e = T.ceil(T.call_extern(T.float32, "log2f", T.call_extern(T.float32, "__fdividef", m, T.float32(448))))
                    e = T.max(T.min(e, T.float32(127)), T.float32(-127))
                    e = T.if_then_else(m == T.float32(0), T.float32(-127), e)
                    inv = T.call_extern(T.float32, "exp2f", -e)
                    q[0] = T.cast(T.min(a0 * inv, T.float32(448)), T.float8_e4m3fn)
                    q[1] = T.cast(T.min(a1 * inv, T.float32(448)), T.float8_e4m3fn)
                    if col < width:
                        row = (n * height + out_y) * width + col
                        for i in T.vectorized(2):
                            out[row * channels + c0 + i] = q[i]
                        if lane == 0:
                            # cuBLAS SWIZZLE_32_4_4 E8M0 layout (128-row x 4-group tiles).
                            r = row % 128
                            scale[((row // 128) * scale_tiles + g // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + g % 4] = (
                                T.cast(e + T.float32(127), T.uint8)
                            )
                for i in T.unroll(8):
                    acc[i] = acc[i + 2]
                acc[8] = T.float32(0)
                acc[9] = T.float32(0)

    return main


@functools.cache
def _depthwise5x5_relu_mxfp8_kernel(batch: int, height: int, width: int, channels: int):
    return _build_depthwise5x5_relu_mxfp8(batch, height, width, channels, _tile_h(height))


@torch.library.custom_op("flashim::depthwise5x5_relu_mxfp8_tilelang", mutates_args=())
def _depthwise5x5_relu_mxfp8(value: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    if value.data_ptr() % 16:  # TMA needs a 16-byte aligned base (checked here: not traceable)
        value = value.clone(memory_format=torch.channels_last)
    batch, channels, height, width = value.shape
    rows = batch * height * width
    scale_tiles = (channels // 32 + 3) // 4
    padded_rows = (rows + 127) // 128 * 128
    quantized = torch.empty((rows, channels), device=value.device, dtype=torch.float8_e4m3fn)
    scale_bits = torch.zeros(padded_rows * scale_tiles * 4, device=value.device, dtype=torch.uint8)
    kernel = _depthwise5x5_relu_mxfp8_kernel(batch, height, width, channels)
    kernel(value.permute(0, 2, 3, 1), weight.reshape(-1), bias, quantized.view(-1), scale_bits)
    return quantized, scale_bits.view(torch.float8_e8m0fnu)


@_depthwise5x5_relu_mxfp8.register_fake
def _(value: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    batch, channels, height, width = value.shape
    rows = batch * height * width
    scale_tiles = (channels // 32 + 3) // 4
    padded_rows = (rows + 127) // 128 * 128
    return (
        value.new_empty((rows, channels), dtype=torch.float8_e4m3fn),
        value.new_empty((padded_rows * scale_tiles * 4,), dtype=torch.float8_e8m0fnu),
    )


def depthwise5x5_relu_mxfp8(value: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """``value`` channels-last BF16 ``[B, C, H, W]`` (C % 64 == 0), ``weight`` BF16 ``[25, C]``,
    ``bias`` FP32 ``[C]``. Returns E4M3 ``[B*H*W, C]`` and swizzled E8M0 scales."""
    return _depthwise5x5_relu_mxfp8(value, weight, bias)


@tilelang.jit
def _build_depthwise5x5_relu_int8(batch: int, height: int, width: int, channels: int, tile_h: int):
    groups = channels // 32
    rows = batch * height * width
    block_groups = groups // GROUPS_PER_BLOCK

    @T.prim_func
    def main(
        x: T.Tensor((batch, height, width, channels), T.bfloat16),
        weight: T.Tensor((25 * channels,), T.bfloat16),
        bias: T.Tensor((channels,), T.float32),
        scale: T.Tensor((1,), T.float32),
        out: T.Tensor((rows * channels,), T.int8),
    ):
        with T.Kernel(
            T.ceildiv(width, COLS), T.ceildiv(height, tile_h), batch * block_groups,
            threads=16 * COLS * GROUPS_PER_BLOCK,
        ) as (bx, by, bz):
            T.no_set_max_nreg()
            T.import_source(_RELEASE_READS_SRC)
            tx = T.get_thread_binding()
            lane = tx % 16
            tcol = tx // (16 * GROUPS_PER_BLOCK)
            gi = (tx // 16) % GROUPS_PER_BLOCK
            col = bx * COLS + tcol
            n = bz // block_groups
            g0 = (bz % block_groups) * GROUPS_PER_BLOCK
            g = g0 + gi
            c0 = g * 32 + lane * 2
            w = T.alloc_local((50,), T.float32)
            acc = T.alloc_local((10,), T.float32)
            raw = T.alloc_local((10,), T.bfloat16)
            b = T.alloc_local((2,), T.float32)
            q = T.alloc_local((2,), T.int8)
            row_s = T.alloc_shared((COLS + 4, 32 * GROUPS_PER_BLOCK), T.bfloat16)
            inv_scale = T.float32(1) / scale[0]
            for k in T.unroll(25):
                for i in T.unroll(2):
                    w[k * 2 + i] = T.cast(weight[k * channels + c0 + i], T.float32)
            for i in T.unroll(2):
                b[i] = bias[c0 + i]
            for i in T.unroll(10):
                acc[i] = T.float32(0)
            for step in T.Pipelined(tile_h + 4, num_stages=STAGES):
                T.copy(
                    x[n, by * tile_h - 2 + step, bx * COLS - 2 : bx * COLS + COLS + 2,
                      g0 * 32 : g0 * 32 + 32 * GROUPS_PER_BLOCK],
                    row_s,
                )
                for kx in T.unroll(5):
                    for i in T.vectorized(2):
                        raw[kx * 2 + i] = row_s[tcol + kx, gi * 32 + lane * 2 + i]
                T.call_extern("handle", "flashim_release_smem_reads", T.access_ptr(row_s[0, 0], "r"))
                for kx in T.unroll(5):
                    for ky in T.unroll(5):
                        for i in T.unroll(2):
                            acc[(4 - ky) * 2 + i] = (
                                acc[(4 - ky) * 2 + i] + T.cast(raw[kx * 2 + i], T.float32) * w[(ky * 5 + kx) * 2 + i]
                            )
                out_y = by * tile_h - 4 + step
                if step >= 4 and out_y < height:
                    for i in T.unroll(2):
                        # ReLU, round to BF16 (as the BF16 conv would), static INT8: rint(x / scale), +-127
                        a = T.cast(T.cast(T.max(acc[i] + b[i], T.float32(0)), T.bfloat16), T.float32)
                        r = T.call_extern(T.float32, "rintf", a * inv_scale)
                        q[i] = T.cast(T.max(T.min(r, T.float32(127)), T.float32(-127)), T.int8)
                    if col < width:
                        row = (n * height + out_y) * width + col
                        for i in T.vectorized(2):
                            out[row * channels + c0 + i] = q[i]
                for i in T.unroll(8):
                    acc[i] = acc[i + 2]
                acc[8] = T.float32(0)
                acc[9] = T.float32(0)

    return main


@functools.cache
def _depthwise5x5_relu_int8_kernel(batch: int, height: int, width: int, channels: int):
    return _build_depthwise5x5_relu_int8(batch, height, width, channels, _tile_h(height))


@torch.library.custom_op("flashim::depthwise5x5_relu_int8_tilelang", mutates_args=())
def _depthwise5x5_relu_int8(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    if value.data_ptr() % 16:
        value = value.clone(memory_format=torch.channels_last)
    batch, channels, height, width = value.shape
    quantized = torch.empty((batch * height * width, channels), device=value.device, dtype=torch.int8)
    kernel = _depthwise5x5_relu_int8_kernel(batch, height, width, channels)
    kernel(value.permute(0, 2, 3, 1), weight.reshape(-1), bias, scale.reshape(1), quantized.view(-1))
    return quantized


@_depthwise5x5_relu_int8.register_fake
def _(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    batch, channels, height, width = value.shape
    return value.new_empty((batch * height * width, channels), dtype=torch.int8)


def depthwise5x5_relu_int8(value: Tensor, weight: Tensor, bias: Tensor, scale: Tensor) -> Tensor:
    """Same operands as the Triton kernel (``value`` channels-last BF16, ``C % 64 == 0``); int8 ``[B*H*W, C]``."""
    return _depthwise5x5_relu_int8(value, weight, bias, scale)
