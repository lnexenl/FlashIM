"""GEMM kernels: FP8 (E4M3) GEMM with block-32 scales for shapes cuBLAS MXFP8 cannot take,
and a small-N BF16 -> FP32 linear."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.backends.triton.quantize import gelu_quantize, to_e4m3
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature, platform_for
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight


def _is_sm89(device: torch.device) -> bool:
    """Launch configs tuned on Ada (an L20) apply to that architecture only."""
    return platform_for(device).capability == (8, 9)


def _is_sm90(device: torch.device) -> bool:
    """Hopper (measured on an H20). Triton lowers ``tl.dot`` tiles of 64+ rows to wgmma there and smaller ones to
    mma.sync; the two E4M3 MMAs round differently (wgmma differs from the exactly rounded BF16 result in 2.6% of a
    256 -> 1024 linear's outputs, mma.sync in 4 of 4.2M), so FP8 kernels are bit-identical only within a row class."""
    return platform_for(device).capability == (9, 0)


@triton.jit
def _fp8_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    bias_ptr,
    out_ptr,
    rows,
    N,
    K: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Static-scale activation x per-row-block weight FP8 GEMM, BF16 output."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        for k_tile in tl.static_range(BLOCK_K // 32):
            offs_k = k0 + k_tile * 32 + tl.arange(0, 32)
            kb = k0 // 32 + k_tile
            a_bits = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_m[:, None], other=0.0)
            a_scale = tl.exp2(tl.load(a_scale_ptr + kb).to(tl.float32))
            b_bits = tl.load(b_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_n[:, None], other=0.0)
            b_scale_exp = tl.load(b_scale_ptr + offs_n * K_BLOCKS + kb, mask=mask_n, other=-127)
            b_scale = tl.exp2(b_scale_exp.to(tl.float32))
            accumulator += tl.dot(a_bits, tl.trans(b_bits)) * (a_scale * b_scale[None, :])
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _gemm_config(rows: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    """Shape-deterministic launch config (RoMaV2 ``select_gemm_config``)."""
    block_k = 32 if rows <= 1024 else (64 if k % 64 == 0 else 32)
    if n >= 128 and rows >= 64:
        block_m = 64
    elif rows >= 128:
        block_m = 128
    elif rows >= 64:
        block_m = 64
    elif rows >= 32:
        block_m = 32
    else:
        block_m = 16
    block_n = 128 if n >= 128 else (64 if n >= 64 else 32)
    return block_m, block_n, block_k, 4, 4 if k >= 2048 else 3


def _gemm_config_sm89(rows: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    """Ada (measured on an L20, where this kernel is the main MXFP8 linear, over block_m {32, 64, 128} x block_n
    {64, 128, 256} x block_k {32, 64, 128} x 4 / 8 warps x 2-4 stages; every config is bit-identical): 128 x 64 tiles
    with one 32-wide K block per iteration win on the transformer / conv-as-GEMM shapes, e.g. 2505 x 4096 -> 1024
    0.181 -> 0.148 ms, 5377 x 4096 -> 1024 0.354 -> 0.280, 5000 x 3072 -> 768 0.188 -> 0.152, 16464 x 4608 -> 512
    0.515 -> 0.440, and 1.07-1.09x with N = 3072 / 4096. An activation matrix far larger than L2 (65856 x 2304
    -> 256) loses with 64-wide N tiles (0.531 ms: it is re-read per N tile) and keeps 64 x 128 x 64 with 3 stages
    (0.482 -> 0.439 ms). N = 256 at 16500 rows is flat, and smaller shapes were not measured: both keep
    :func:`_gemm_config`."""
    if rows * k >= 1 << 27 and n >= 128:
        return 64, 128, 64 if k % 64 == 0 else 32, 4, 3
    if rows >= 2048 and n >= 512:
        return 128, 64, 32, 4, 3
    return _gemm_config(rows, k, n)


def _gemm_config_sm90(rows: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    """Hopper (measured on an H20 over block_m {32, 64, 128} x block_n {64, 128, 256} x block_k {32, 64}): the 64 x 128
    wgmma tiles of :func:`_gemm_config` with one 32-wide K block per iteration and 3 stages, bit-identical to its
    64-wide K blocks: 2505 x 1024 -> 4096 0.135 -> 0.120 ms, 5377 x 4096 -> 1024 0.340 -> 0.316, 5000 x 3072 -> 768
    0.166 -> 0.134, 16464 x 4608 -> 512 0.501 -> 0.372, 65856 x 2304 -> 256 0.406 -> 0.354, 2505 x 4096 -> 1024 0.223
    -> 0.209 (64-wide K with 3 stages: 0.185 there, slower on the others). The Ada pick (128 x 64) loses here (0.175 /
    0.382 / 0.214 / 0.586 / 0.587 ms), and 32-row tiles (mma.sync) are 2-2.7x slower. Up to 1024 rows is flat and
    N < 128 was not measured: both keep :func:`_gemm_config`."""
    block_m, block_n, block_k, num_warps, num_stages = _gemm_config(rows, k, n)
    if rows > 1024 and n >= 128:
        return block_m, block_n, 32, num_warps, 3
    return block_m, block_n, block_k, num_warps, num_stages


_GEMM_CONFIGS = {(8, 9): _gemm_config_sm89, (9, 0): _gemm_config_sm90}


def fp8_gemm_bf16(
    activation: QuantizedTensor, weight: QuantizedWeight, bias: Tensor | None, config: tuple | None = None
) -> Tensor:
    """``[rows, K] x [N, K]^T`` with static activation scales; padded BF16 output."""
    matrix = activation.matrix().contiguous()
    rows = matrix.shape[0]
    n, k = weight.out_features, weight.in_features
    out = torch.empty(rows, n, device=matrix.device, dtype=torch.bfloat16)
    if config is None:
        config = _GEMM_CONFIGS.get(platform_for(matrix.device).capability, _gemm_config)(rows, k, n)
    block_m, block_n, block_k, num_warps, num_stages = config
    _fp8_gemm_kernel[(triton.cdiv(rows, block_m) * triton.cdiv(n, block_n),)](
        matrix,
        activation.scale_exponent,
        weight.values,
        weight.scale_exponent,
        bias if bias is not None else out,
        out,
        rows,
        n,
        K=k,
        K_BLOCKS=k // 32,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# -- dynamic block-scaled MXFP8 GEMM (F.scaled_mm operands) without block-scaled MMA --------------------


@triton.jit
def _swizzled_scale(scale_ptr, row, group, mask, SCALE_TILES: tl.constexpr):
    """``2**e`` of (``row``, 32-block ``group``) read from cuBLAS' ``SWIZZLE_32_4_4`` E8M0 bytes
    (``mx.swizzle_scales``; the layout ``depthwise5x5_relu_mxfp8`` writes)."""
    in_tile = row % 128
    offset = ((row // 128) * SCALE_TILES + group // 4) * 512 + (in_tile % 32) * 16 + (in_tile // 32) * 4 + group % 4
    return tl.exp2(tl.load(scale_ptr + offset, mask=mask, other=0).to(tl.float32) - 127.0)


@triton.jit
def _mxfp8_scaled_mm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    bias_ptr,
    out_ptr,
    rows,
    N,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``[rows, K] x [N, K]^T`` with one E8M0 scale per (row, 32 elements) on both operands, BF16 output: one E4M3
    ``mma.sync`` per 32-wide block of ``K``, its FP32 result scaled by the outer product of the two blocks' scales
    (sm_89 / sm_90 have E4M3 MMA but no block-scaled one)."""
    GROUPS: tl.constexpr = K // 32
    SCALE_TILES: tl.constexpr = (GROUPS + 3) // 4
    offs_m = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < N
    k32 = tl.arange(0, 32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for group in range(GROUPS):
        offs_k = group * 32 + k32
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptr + offs_n[None, :] * K + offs_k[:, None], mask=mask_n[None, :], other=0.0)
        a_scale = _swizzled_scale(a_scale_ptr, offs_m, group, mask_m, SCALE_TILES)
        b_scale = _swizzled_scale(b_scale_ptr, offs_n, group, mask_n, SCALE_TILES)
        accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale[None, :])
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _scaled_mm_config(k: int, n: int) -> dict[str, int]:
    """Measured on an L20 on RoMaV2's refiner pointwise calls (``K = N``; block_m / block_n in {32, 64, 128}, 4 / 8
    warps, 1-3 stages, interleaved with the previous 64 x 64, 2-stage tiles; all bit-identical): 128 x 128 with 3
    stages is best or tied everywhere, and 1.4-1.9x faster at K = 128 (262144 rows 0.141 -> 0.100 ms, 160000 rows
    0.097 -> 0.050 ms: the 4-iteration K loop does not amortize 64 x 64 tiles); K = 512 (65536 rows 0.225 ms) and
    K = 32 are flat. A BF16 cuBLAS ``addmm`` of the K = 512 / 128 shapes takes about 0.40 / 0.25 ms."""
    return {"block_m": 128, "block_n": min(128, max(16, triton.next_power_of_2(n))), "num_warps": 4, "num_stages": 3}


def mxfp8_scaled_mm(
    a: Tensor, b_t: Tensor, a_scale: Tensor, b_scale: Tensor, bias: Tensor | None = None, *, tile: dict | None = None
) -> Tensor:
    rows, k = a.shape
    n = b_t.shape[1]
    out = torch.empty((rows, n), device=a.device, dtype=torch.bfloat16)
    tile = tile or _scaled_mm_config(k, n)
    block_n = tile["block_n"]
    _mxfp8_scaled_mm_kernel[(triton.cdiv(rows, tile["block_m"]), triton.cdiv(n, block_n))](
        a,
        a_scale.view(torch.uint8),
        b_t.t(),
        b_scale.view(torch.uint8),
        bias if bias is not None else out,
        out,
        rows,
        n,
        K=k,
        HAS_BIAS=bias is not None,
        BLOCK_M=tile["block_m"],
        BLOCK_N=block_n,
        num_warps=tile["num_warps"],
        num_stages=tile["num_stages"],
    )
    return out


@triton.jit
def _linear_fp32_small_n_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, rows, K: tl.constexpr, P: tl.constexpr, BM: tl.constexpr
):
    """``out = x @ w.T + b`` for small ``P``: BF16 ``x`` tile in registers, one exact-FP32
    FMA reduction per output channel (cuBLAS's FP32 GEMM is inefficient for ``N ~ 12``)."""
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = row < rows
    k = tl.arange(0, K)
    x = tl.load(x_ptr + row[:, None] * K + k[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    for p in tl.static_range(P):
        out = tl.sum(x * tl.load(w_ptr + p * K + k)[None, :], axis=1) + tl.load(b_ptr + p)
        tl.store(out_ptr + row * P + p, out, mask=valid)


def linear_fp32_small_n(x: Tensor, weight: Tensor, bias: Tensor, *, block_m: int = 128, num_warps: int = 4) -> Tensor:
    *lead, k = x.shape
    p = weight.shape[0]
    rows = x.numel() // k
    out = torch.empty((*lead, p), device=x.device, dtype=torch.float32)
    _linear_fp32_small_n_kernel[(triton.cdiv(rows, block_m),)](
        x, weight.float().contiguous(), bias.float().contiguous(), out, rows, K=k, P=p, BM=block_m,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _int8_gemm_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    out_ptr,
    rows,
    N,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """``int8[M,K] x int8[N,K]^T`` with INT32 accumulation; epilogue ``acc * (a_scale * w_scale[n]) + bias`` (then ReLU
    with ``RELU``: it commutes with the BF16 rounding, so it equals a separate ReLU pass bit for bit)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * num_pid_n)
    first_m = group * GROUP_M
    group_size = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size)
    pid_n = (pid % (GROUP_M * num_pid_n)) // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < rows
    mask_n = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = offs_k < K - k0
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_n[None, :] & k_mask[:, None], other=0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    scale = tl.load(a_scale_ptr) * tl.load(w_scale_ptr + offs_n, mask=mask_n, other=0.0)
    if HAS_BIAS:  # explicit FMA: the compiler contracted `acc * scale + bias` for most tile configs but not all,
        # so the output bits depended on which config autotuning picked; one rounding for every config instead
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = tl.fma(accumulator.to(tl.float32), scale[None, :], bias[None, :])
    else:
        out = accumulator.to(tl.float32) * scale[None, :]
    if RELU:
        out = tl.maximum(out, 0.0)
    tl.store(
        out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :],
        out.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _int8_gemm_config(
    rows: int, n: int, small_tiles: bool = False, four_warps: bool = False
) -> tuple[int, int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages). ``small_tiles`` (sm_89, sm_90): 4-warp 64-row tiles
    measured faster than 128 x 128 with 8 warps on an L20 (exact INT32 accumulation: bit-identical), e.g. 2505 x 1024
    -> 3072 0.088 -> 0.077 ms, 2505 x 4096 -> 1024 0.114 -> 0.103, 131072 x 512 -> 512 0.341 -> 0.320 (64 x 128); N <=
    256 is flat. On an H20 the same picks win too: 0.079 -> 0.065, 0.122 -> 0.091, 0.328 -> 0.267 ms, and 16464 x 4608
    -> 512 0.314 -> 0.286. ``four_warps`` (sm_87): on a Jetson AGX Orin the 128 x 128 tile wins, with 4 warps instead
    of 8 (the 64-row picks above are 7-25% slower there): 2505 x 1024 -> 1024 0.126 -> 0.121 ms, 2505 x 1024 -> 4096
    0.489 -> 0.451, 2505 x 4096 -> 1024 0.410 -> 0.368, 10020 x 1024 -> 1024 0.480 -> 0.440, 131072 x 512 -> 512 1.734 ->
    1.693."""
    if small_tiles and rows >= 2048 and n >= 512:
        return 64, 128 if rows >= 32768 else 64, 64, 4, 3
    if rows >= 512:
        return 128, 128, 64, 4 if four_warps else 8, 3
    if rows >= 128:
        return 64, 128, 64, 4, 4
    return 32, 64, 128, 4, 3


def int8_gemm_bf16(
    a: Tensor, weight: Tensor, a_scale: Tensor, w_scale: Tensor, bias: Tensor | None, config=None, *, relu: bool = False
) -> Tensor:
    """``[rows, K] int8 x [N, K] int8^T`` -> BF16 ``[rows, N]`` with the dequantizing epilogue fused."""
    rows, k = a.shape
    n = weight.shape[0]
    out = torch.empty(rows, n, device=a.device, dtype=torch.bfloat16)
    block_m, block_n, block_k, num_warps, num_stages = config or _int8_gemm_config(
        rows, n, _is_sm89(a.device) or _is_sm90(a.device), platform_for(a.device).capability == (8, 7)
    )
    _int8_gemm_kernel[(triton.cdiv(rows, block_m) * triton.cdiv(n, block_n),)](
        a,
        weight,
        a_scale,
        w_scale,
        bias if bias is not None else out,
        out,
        rows,
        n,
        K=k,
        HAS_BIAS=bias is not None,
        RELU=relu,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit
def _linear_fp32_split_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, rows, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """``out = x @ w.T + b`` with BF16 ``x`` and FP32 ``w`` on tensor cores: ``w = hi + lo`` (two BF16
    halves, ~16 mantissa bits), two BF16 ``tl.dot`` per tile, FP32 accumulation and output."""
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    mask_m = offs_m < rows
    mask_n = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * K + offs_k[None, :], mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * K + offs_k[:, None], mask=mask_n[None, :], other=0.0)
        hi = w.to(tl.bfloat16)
        lo = (w - hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(a, hi, acc)
        acc = tl.dot(a, lo, acc)
    acc += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


def linear_fp32_split(x: Tensor, weight: Tensor, bias: Tensor, *, bm: int = 64, bn: int = 64, num_warps: int = 4) -> Tensor:
    *lead, k = x.shape
    n = weight.shape[0]
    rows = x.numel() // k
    out = torch.empty((*lead, n), device=x.device, dtype=torch.float32)
    _linear_fp32_split_kernel[(triton.cdiv(rows, bm), triton.cdiv(n, bn))](
        x, weight.float().contiguous(), bias.float().contiguous(), out, rows, n, K=k, BM=bm, BN=bn,
        BK=min(64, k), num_warps=num_warps,
    )
    return out


# -- fused MXFP8 GELU MLP: quantize(GELU(x W1^T + b1)) W2^T, hidden kept on chip ------------------------


@triton.jit
def _mxfp8_gelu_mlp_kernel(
    x_ptr,
    x_exponent_ptr,
    w1_ptr,
    w1_scale_ptr,
    b1_ptr,
    hidden_exponent_ptr,
    w2_ptr,
    w2_scale_ptr,
    out_ptr,
    rows,
    K1: tl.constexpr,
    HIDDEN: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One ``BLOCK_M`` row tile: for each ``BLOCK_H`` hidden chunk, ``h = x W1_chunk^T`` (block-scaled MMA),
    rounded to BF16, ``+ b1``, exact GELU, rounded to BF16, statically quantized to E4M3 with the hidden
    exponents, then accumulated into ``h W2_chunk^T``. BF16 output without W2's bias.

    Rounding points match cuBLAS linear1 (BF16 output) -> ``gelu_quantize(input_bias=b1)`` -> cuBLAS linear2.
    """
    K1_GROUPS: tl.constexpr = K1 // 32
    H_GROUPS: tl.constexpr = BLOCK_H // 32
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    k1 = tl.arange(0, K1)
    n2 = tl.arange(0, N2)
    hcol = tl.arange(0, BLOCK_H)
    x = tl.load(x_ptr + offs_m.to(tl.int64)[:, None] * K1 + k1[None, :], mask=mask_m[:, None], other=0.0)
    x_scale = (tl.load(x_exponent_ptr + tl.arange(0, K1_GROUPS)).to(tl.int32) + 127).to(tl.uint8)
    x_scale = tl.broadcast_to(x_scale[None, :], (BLOCK_M, K1_GROUPS))
    acc = tl.zeros((BLOCK_M, N2), dtype=tl.float32)
    for h0 in range(0, HIDDEN, BLOCK_H):
        # W1 rows [h0, h0 + BLOCK_H) as the [K1, BLOCK_H] operand, scales [BLOCK_H, K1/32].
        w1 = tl.load(w1_ptr + (h0 + hcol)[None, :] * K1 + k1[:, None])
        w1_scale = tl.load(w1_scale_ptr + (h0 + hcol)[:, None] * K1_GROUPS + tl.arange(0, K1_GROUPS)[None, :])
        hidden = tl.dot_scaled(x, x_scale, "e4m3", w1, w1_scale, "e4m3")
        hidden = hidden.to(tl.bfloat16).to(tl.float32) + tl.load(b1_ptr + h0 + hcol).to(tl.float32)[None, :]
        hidden = 0.5 * hidden * (1.0 + tl.math.erf(hidden * 0.7071067811865476))
        hidden = hidden.to(tl.bfloat16).to(tl.float32)
        exponent = tl.load(hidden_exponent_ptr + h0 // 32 + tl.arange(0, H_GROUPS)).to(tl.float32)
        blocks = tl.reshape(hidden, (BLOCK_M, H_GROUPS, 32)) * tl.exp2(-exponent)[None, :, None]
        quantized = tl.reshape(tl.maximum(tl.minimum(blocks, 448.0), -448.0), (BLOCK_M, BLOCK_H)).to(tl.float8e4nv)
        h_scale = tl.broadcast_to((exponent.to(tl.int32) + 127).to(tl.uint8)[None, :], (BLOCK_M, H_GROUPS))
        # W2 columns [h0, h0 + BLOCK_H) as the [BLOCK_H, N2] operand, scales [N2, BLOCK_H/32].
        w2 = tl.load(w2_ptr + n2[None, :] * HIDDEN + (h0 + hcol)[:, None])
        w2_scale = tl.load(
            w2_scale_ptr + n2[:, None] * (HIDDEN // 32) + h0 // 32 + tl.arange(0, H_GROUPS)[None, :]
        )
        acc = tl.dot_scaled(quantized, h_scale, "e4m3", w2, w2_scale, "e4m3", acc)
    out_ptrs = out_ptr + offs_m.to(tl.int64)[:, None] * N2 + n2[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


MLP_TILE = {"block_m": 64, "block_h": 64, "num_warps": 4, "num_stages": 1}


def mxfp8_gelu_mlp(
    x: QuantizedTensor, w1: QuantizedWeight, b1: Tensor, hidden_exponent: Tensor, w2: QuantizedWeight
) -> Tensor:
    rows, k1 = x.rows, x.channels
    hidden, n2 = w1.out_features, w2.out_features
    out = torch.empty((rows, n2), device=x.device, dtype=torch.bfloat16)
    tile = MLP_TILE
    _mxfp8_gelu_mlp_kernel[(triton.cdiv(rows, tile["block_m"]),)](
        x.matrix(),
        x.scale_exponent,
        w1.values,
        (w1.scale_exponent.to(torch.int32) + 127).to(torch.uint8),
        b1.float(),
        hidden_exponent,
        w2.values,
        (w2.scale_exponent.to(torch.int32) + 127).to(torch.uint8),
        out,
        rows,
        K1=k1,
        HIDDEN=hidden,
        N2=n2,
        BLOCK_M=tile["block_m"],
        BLOCK_H=tile["block_h"],
        num_warps=tile["num_warps"],
        num_stages=tile["num_stages"],
    )
    return out


@triton.jit
def _mxfp8_gelu_mlp_mma_kernel(
    x_ptr,
    x_exponent_ptr,
    w1_ptr,
    w1_exponent_ptr,
    b1_ptr,
    hidden_exponent_ptr,
    w2_ptr,
    w2_exponent_ptr,
    out_ptr,
    rows,
    K1: tl.constexpr,
    HIDDEN: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """``_mxfp8_gelu_mlp_kernel`` without block-scaled MMA (sm_89 / sm_90: E4M3 ``mma.sync`` only): the hidden
    chunk is one 32-wide scale block, every E4M3 MMA covers exactly one block of ``K`` and its FP32 result is scaled
    by the two blocks' exponents. Same rounding points (BF16 hidden, ``+ b1``, exact GELU, BF16, static E4M3)."""
    K1_GROUPS: tl.constexpr = K1 // 32
    H_GROUPS: tl.constexpr = HIDDEN // 32
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    k32 = tl.arange(0, 32)
    n2 = tl.arange(0, N2)
    x_rows = x_ptr + offs_m.to(tl.int64)[:, None] * K1
    acc = tl.zeros((BLOCK_M, N2), dtype=tl.float32)
    for h0 in range(0, HIDDEN, 32):
        hcol = h0 + k32
        hidden = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
        for group in tl.static_range(K1_GROUPS):
            x = tl.load(x_rows + (group * 32 + k32)[None, :], mask=mask_m[:, None], other=0.0)
            w1 = tl.load(w1_ptr + hcol[None, :] * K1 + (group * 32 + k32)[:, None])
            x_scale = tl.exp2(tl.load(x_exponent_ptr + group).to(tl.float32))
            w1_scale = tl.exp2(tl.load(w1_exponent_ptr + hcol * K1_GROUPS + group).to(tl.float32))
            hidden += tl.dot(x, w1) * (x_scale * w1_scale)[None, :]
        hidden = hidden.to(tl.bfloat16).to(tl.float32) + tl.load(b1_ptr + hcol).to(tl.float32)[None, :]
        hidden = 0.5 * hidden * (1.0 + tl.math.erf(hidden * 0.7071067811865476))
        hidden = hidden.to(tl.bfloat16).to(tl.float32)
        exponent = tl.load(hidden_exponent_ptr + h0 // 32).to(tl.float32)
        quantized = to_e4m3(tl.maximum(tl.minimum(hidden * tl.exp2(-exponent), 448.0), -448.0))
        w2 = tl.load(w2_ptr + n2[None, :] * HIDDEN + hcol[:, None])
        w2_scale = tl.exp2(tl.load(w2_exponent_ptr + n2 * H_GROUPS + h0 // 32).to(tl.float32))
        acc += tl.dot(quantized, w2) * (tl.exp2(exponent) * w2_scale)[None, :]
    out_ptrs = out_ptr + offs_m.to(tl.int64)[:, None] * N2 + n2[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


# K1 = 256, hidden = 1024, N2 = 256 on an L20: 0.044 / 0.078 / 0.265 ms at 2048 / 8192 / 30000 rows, against 0.100 /
# 0.099 / 0.282 ms for mxfp8_linear -> gelu_quantize -> mxfp8_linear (bit-identical output) and 0.22 / 0.70 / 4.4 ms
# for the FP32 reference. block_m = 64 ties at 30000 rows and loses below (0.063 / 0.096 ms); 8 warps is 2x slower.
MLP_MMA_TILE = {"block_m": 32, "num_warps": 4, "num_stages": 1}


def mxfp8_gelu_mlp_mma(
    x: QuantizedTensor, w1: QuantizedWeight, b1: Tensor, hidden_exponent: Tensor, w2: QuantizedWeight
) -> Tensor:
    rows, k1 = x.rows, x.channels
    hidden, n2 = w1.out_features, w2.out_features
    out = torch.empty((rows, n2), device=x.device, dtype=torch.bfloat16)
    tile = MLP_MMA_TILE
    _mxfp8_gelu_mlp_mma_kernel[(triton.cdiv(rows, tile["block_m"]),)](
        x.matrix(),
        x.scale_exponent,
        w1.values,
        w1.scale_exponent,
        b1.float(),
        hidden_exponent,
        w2.values,
        w2.scale_exponent,
        out,
        rows,
        K1=k1,
        HIDDEN=hidden,
        N2=n2,
        BLOCK_M=tile["block_m"],
        num_warps=tile["num_warps"],
        num_stages=tile["num_stages"],
    )
    return out

# -- registrations ---------------------------------------------------------------------------------------

# Static MXFP8 linear for shapes cuBLAS block-scaled MXFP8 cannot take (and pre-Blackwell FP8 GPUs).
@ops.mxfp8_linear.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3])
def _mxfp8_linear_impl(activation: QuantizedTensor, bias: Tensor | None, weight: QuantizedWeight) -> Tensor:
    return fp8_gemm_bf16(activation, weight, None if bias is None else bias.float())


def _scaled_mm_eligible(a, b_t, a_scale, b_scale, bias=None) -> bool:
    return (
        a.ndim == 2
        and b_t.ndim == 2
        and a.dtype == torch.float8_e4m3fn
        and b_t.dtype == torch.float8_e4m3fn
        and a.shape[1] == b_t.shape[0]
        and a.shape[1] % 32 == 0
        and a.is_contiguous()
        and b_t.t().is_contiguous()  # the weight as stored, [N, K]
        and a_scale.is_contiguous()
        and b_scale.is_contiguous()
    )


# FP8 GPUs without cuBLAS block-scaled MXFP8 (sm_89, sm_90); cuBLAS has priority where it exists.
@ops.mxfp8_scaled_mm.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3], when=_scaled_mm_eligible
)
def _mxfp8_scaled_mm_impl(a, b_t, a_scale, b_scale, bias=None):
    return mxfp8_scaled_mm(a, b_t, a_scale, b_scale, bias)


# The fused-epilogue Triton GEMM beats cuBLAS ``_int_mm`` + separate dequant passes at every shape.
@ops.int8_linear.register(
    "triton", device=DeviceSpec("cuda", min_capability=(7, 5)), requires=[Feature.TRITON], priority=1
)
def _int8_linear_impl(a, weight, a_scale, w_scale, bias, *, relu: bool = False):
    return int8_gemm_bf16(a.contiguous(), weight, a_scale, w_scale, bias, relu=relu)


def _linear_fp32_small_n_eligible(x: Tensor, weight: Tensor, bias: Tensor) -> bool:
    k = x.shape[-1]
    return (
        x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.shape[0] <= 16
        and weight.shape[1] == k
        and k in (16, 32, 64, 128)
        and x.numel() < 2**31
    )


ops.linear_fp32.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_linear_fp32_small_n_eligible
)(linear_fp32_small_n)


def _linear_fp32_split_eligible(x: Tensor, weight: Tensor, bias: Tensor) -> bool:
    return (
        x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.shape[0] > 16
        and weight.shape[1] == x.shape[-1]
        and x.shape[-1] % 32 == 0
        and x.numel() < 2**31
    )


# Tensor-core path: exact BF16 input, weight as two BF16 halves (~1e-5 relative error vs FP32).
ops.linear_fp32.register(
    "triton_split", device=DeviceSpec("cuda", min_capability=(8, 0)), requires=[Feature.TRITON],
    when=_linear_fp32_split_eligible,
)(linear_fp32_split)


def _mlp_eligible(x, w1, b1, hidden_exponent, w2) -> bool:
    k1, hidden, n2 = x.channels, w1.out_features, w2.out_features
    return (
        w1.in_features == k1
        and w2.in_features == hidden
        and k1 in (32, 64, 128, 256)
        and n2 in (32, 64, 128, 256)
        and hidden % 64 == 0
    )


@ops.mxfp8_gelu_mlp.register(
    "triton", device=DeviceSpec("cuda", min_capability=(10, 0)), requires=[Feature.TRITON], when=_mlp_eligible
)
def _mxfp8_gelu_mlp_impl(x, w1, b1, hidden_exponent, w2):
    return mxfp8_gelu_mlp(x, w1, b1, hidden_exponent.to(x.device, torch.int8), w2)


# FP8 GPUs without block-scaled MMA (sm_89, sm_90); the narrower registrations around it win where they apply.
@ops.mxfp8_gelu_mlp.register(
    "triton_mma", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3], when=_mlp_eligible
)
def _mxfp8_gelu_mlp_mma_impl(x, w1, b1, hidden_exponent, w2):
    return mxfp8_gelu_mlp_mma(x, w1, b1, hidden_exponent.to(x.device, torch.int8), w2)


# Hopper: the three per-op kernels. Their linears run 64-row wgmma tiles, the fused kernel's 32-row tiles mma.sync (see
# ``_is_sm90``; the outputs differ accordingly). H20, K1 = 256, hidden = 1024, N2 = 256 at 2048 / 8192 / 30000 rows:
# 0.073 / 0.098 / 0.308 ms against 0.070 / 0.192 / 0.563 ms for ``triton_mma`` (0.119 / 0.141 / 0.524 ms with 64-row tiles,
# then bit-identical to this chain).
@ops.mxfp8_gelu_mlp.register(
    "triton_chain", device=DeviceSpec("cuda", min_capability=(9, 0), max_capability=(9, 0)),
    requires=[Feature.TRITON, Feature.FP8_E4M3], when=_mlp_eligible,
)
def _mxfp8_gelu_mlp_chain_impl(x, w1, b1, hidden_exponent, w2):
    hidden = fp8_gemm_bf16(x, w1, None)
    activated = gelu_quantize(hidden, hidden_exponent.to(x.device, torch.int8), "none", input_bias=b1)
    return fp8_gemm_bf16(activated, w2, None)
