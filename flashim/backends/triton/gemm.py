"""GEMM kernels: FP8 (E4M3) GEMM with block-32 scales for shapes cuBLAS MXFP8 cannot take,
and a small-N BF16 -> FP32 linear."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight


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


def fp8_gemm_bf16(activation: QuantizedTensor, weight: QuantizedWeight, bias: Tensor | None) -> Tensor:
    """``[rows, K] x [N, K]^T`` with static activation scales; padded BF16 output."""
    matrix = activation.matrix().contiguous()
    rows = matrix.shape[0]
    n, k = weight.out_features, weight.in_features
    out = torch.empty(rows, n, device=matrix.device, dtype=torch.bfloat16)
    block_m, block_n, block_k, num_warps, num_stages = _gemm_config(rows, k, n)
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


def _int8_gemm_config(rows: int, n: int) -> tuple[int, int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)."""
    if rows >= 512:
        return 128, 128, 64, 8, 3
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
    block_m, block_n, block_k, num_warps, num_stages = config or _int8_gemm_config(rows, n)
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

# -- registrations ---------------------------------------------------------------------------------------

# Static MXFP8 linear for shapes cuBLAS block-scaled MXFP8 cannot take (and pre-Blackwell FP8 GPUs).
@ops.mxfp8_linear.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON, Feature.FP8_E4M3])
def _mxfp8_linear_impl(activation: QuantizedTensor, bias: Tensor | None, weight: QuantizedWeight) -> Tensor:
    return fp8_gemm_bf16(activation, weight, None if bias is None else bias.float())


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
