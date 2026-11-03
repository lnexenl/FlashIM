"""Prototype: per-token (per-row) dynamic INT8 linear for the LoMa extractor's DINOv2 ViT (experiment, not library code).

Row quantize: one program per row, scale = amax / 127, round half to even. GEMM: FlashIM's INT8 Triton kernel with a
per-row ``a_scale`` in the epilogue."""
import torch
import triton
import triton.language as tl
from torch import Tensor, nn
from triton.language.extra import libdevice

from flashim.backends.triton.gemm import _int8_gemm_config, _is_sm89, _is_sm90
from flashim.platforms import platform_for
from flashim.quant.formats import int8

EPS = torch.finfo(torch.float32).eps


@triton.jit
def _row_quant_kernel(x_ptr, q_ptr, s_ptr, K: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row.to(tl.int64) * K + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) / 127.0, EPS)
    q = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(x, scale)), -127.0), 127.0)
    tl.store(q_ptr + row.to(tl.int64) * K + offs, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)


def row_quant(x: Tensor) -> tuple[Tensor, Tensor]:
    rows, k = x.shape
    q = torch.empty(rows, k, device=x.device, dtype=torch.int8)
    s = torch.empty(rows, device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(k)
    _row_quant_kernel[(rows,)](x, q, s, K=k, BLOCK=block, EPS=EPS, num_warps=4 if block <= 1024 else 8)
    return q, s


def row_quant_reference(x: Tensor) -> tuple[Tensor, Tensor]:
    s = (x.float().abs().amax(-1) / 127.0).clamp_min(EPS)
    return (x.float() / s[:, None]).round().clamp(-127, 127).to(torch.int8), s


@triton.jit
def _gemm_kernel(a_ptr, b_ptr, a_scale_ptr, w_scale_ptr, bias_ptr, out_ptr, rows, N, K: tl.constexpr,
                 HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_M: tl.constexpr):
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
    scale = (tl.load(a_scale_ptr + offs_m, mask=mask_m, other=0.0)[:, None]
             * tl.load(w_scale_ptr + offs_n, mask=mask_n, other=0.0)[None, :])
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = tl.fma(accumulator.to(tl.float32), scale, bias[None, :])
    else:
        out = accumulator.to(tl.float32) * scale
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], out.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


def gemm(a: Tensor, a_scale: Tensor, weight: Tensor, w_scale: Tensor, bias: Tensor | None) -> Tensor:
    rows, k = a.shape
    n = weight.shape[0]
    out = torch.empty(rows, n, device=a.device, dtype=torch.bfloat16)
    bm, bn, bk, warps, stages = _int8_gemm_config(
        rows, n, _is_sm89(a.device) or _is_sm90(a.device), platform_for(a.device).capability == (8, 7))
    _gemm_kernel[(triton.cdiv(rows, bm) * triton.cdiv(n, bn),)](
        a, weight, a_scale, w_scale, bias if bias is not None else out, out, rows, n, K=k, HAS_BIAS=bias is not None,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=warps, num_stages=stages)
    return out


class DynamicInt8Linear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        w_scale = int8.weight_scales(linear.weight)
        self.register_buffer("weight", int8.quantize_symmetric(linear.weight, w_scale[:, None]).contiguous())
        self.register_buffer("w_scale", w_scale.contiguous())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.float().contiguous())

    def forward(self, x: Tensor) -> Tensor:
        lead, k = x.shape[:-1], x.shape[-1]
        q, s = row_quant(x.reshape(-1, k).contiguous())
        return gemm(q, s, self.weight, self.w_scale, self.bias).reshape(*lead, -1)

    def reference(self, x: Tensor) -> Tensor:
        q, s = row_quant_reference(x.reshape(-1, x.shape[-1]))
        out = (q.double() @ self.weight.double().t()).float() * (s[:, None] * self.w_scale)
        if self.bias is not None:
            out = out + self.bias
        return out.to(torch.bfloat16).reshape(*x.shape[:-1], -1)


TYPES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def quantize_vit(extractor: nn.Module, types=TYPES) -> int:
    """Replace the DINOv2 ViT's BF16 linears (left by the int8 preset) with dynamic INT8 ones."""
    vit = extractor.descriptor.encoder.dinov2
    count = 0
    for block in vit.blocks:
        for t in types:
            parent, leaf = t.split(".")
            module = getattr(block.get_submodule(parent), leaf)
            assert isinstance(module, nn.Linear), type(module)
            setattr(block.get_submodule(parent), leaf, DynamicInt8Linear(module))
            count += 1
    return count
