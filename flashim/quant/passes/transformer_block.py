"""Fuse quantization boundaries into pre-norm transformer blocks (RoMaV2 FastFP8 contract).

For a block ``x + ls1(attn(norm1(x)))`` / ``x + ls2(mlp(norm2(x)))`` whose attention
QKV/proj and MLP fc1/fc2 are static-quantized linears (:class:`StaticMXFP8Linear` or
:class:`StaticInt8Linear`; both expose ``quantize_layernorm`` / ``quantize_gelu`` /
``forward_prequantized``):
- ``norm1`` is fused with QKV's input quantization;
- the residual add + ``norm2`` is fused with fc1's input quantization when the
  residual stream and the attention output share a dtype;
- GELU is fused with fc2's input quantization;
- attention itself stays BF16 SDPA. RoPE uses the reference (eager) math with the
  tables cast to the query dtype.
When dtypes differ (e.g. an FP32 residual stream under autocast) the block falls back
to the unfused path, exactly as RoMaV2 does.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.layers import TransformerBlock
from flashim.quant.methods.int8_linear import StaticInt8Linear
from flashim.quant.methods.linear import StaticMXFP8Linear

STATIC_LINEARS = (StaticMXFP8Linear, StaticInt8Linear)


def is_transformer_block(module: nn.Module) -> bool:
    return isinstance(module, TransformerBlock)


class QuantizedAttention(nn.Module):
    def __init__(self, attn: nn.Module, rope_impl: str | None = "reference") -> None:
        super().__init__()
        self.qkv, self.proj, self.num_heads = attn.qkv, attn.proj, attn.num_heads
        dim = self.proj.in_features
        head_dim = dim // self.num_heads
        if self.qkv.out_features != 3 * dim or dim % self.num_heads:
            raise ValueError("QuantizedAttention requires standard packed QKV")
        if head_dim % 32 or head_dim & (head_dim - 1):
            raise ValueError(f"head_dim must be a power of two divisible by 32, got {head_dim}")
        self.rope_impl = rope_impl
        # ``ops.sdpa`` implementation: "reference" (torch SDPA), "sage" (INT8 QK attention) or None = dispatch.
        self.sdpa_impl: str | None = getattr(attn, "sdpa_impl", "reference")

    def attend(self, qkv: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        batch, tokens, three_c = qkv.shape
        channels = three_c // 3
        q, k, v = qkv.view(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        if rope is not None:
            sin, cos = (t.reshape(-1, t.shape[-1]).to(device=q.device, dtype=q.dtype) for t in rope)
            q, k = ops.rope_qk.get(self.rope_impl)(q, k, sin, cos)
        out = ops.sdpa.get(self.sdpa_impl)(q, k, v)
        return out.transpose(1, 2).reshape(batch, tokens, channels)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        return self.proj(self.attend(self.qkv(x), rope))


class QuantizedTransformerBlock(nn.Module):
    """``fused_epilogues`` (off by default; :func:`enable_fused_epilogues`): every bias and residual add of an MXFP8
    block (LayerScale folded away, :func:`fold_layer_scale`) rides in a neighbouring kernel instead of an elementwise
    pass over ``[tokens, C]``: qkv / fc2 biases in the cuBLAS epilogue, proj's bias and the residual add in the
    ``norm2`` quantize kernel (which also writes the residual stream), fc1's bias in the GELU quantize kernel. Each
    bias is added before the BF16 rounding instead of after it."""

    fused_epilogues = False

    def __init__(self, block: nn.Module, rope_impl: str | None = "reference") -> None:
        super().__init__()
        if not is_transformer_block(block):
            raise TypeError(f"Not a flashim.layers.TransformerBlock: {type(block).__name__}")
        self.norm1, self.ls1 = block.norm1, block.ls1
        self.norm2, self.mlp, self.ls2 = block.norm2, block.mlp, block.ls2
        self.attn = QuantizedAttention(block.attn, rope_impl)

    def fusable_epilogues(self) -> bool:
        linears = (self.attn.qkv, self.attn.proj, self.mlp.fc1, self.mlp.fc2)
        return (
            all(isinstance(m, StaticMXFP8Linear) and m.bias is not None for m in linears)
            and isinstance(self.ls1, nn.Identity)
            and isinstance(self.ls2, nn.Identity)
            and isinstance(self.mlp.act, nn.GELU)
        )

    def _forward_fused(self, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        shape = x.shape
        channels = shape[-1]
        x_flat = x.reshape(-1, channels)
        qkv, proj, fc1, fc2 = self.attn.qkv, self.attn.proj, self.mlp.fc1, self.mlp.fc2
        qkv_out = qkv.matmul(qkv.quantize_layernorm(x_flat, None, self.norm1), bias=True)
        attended = self.attn.attend(qkv_out.to(x.dtype).reshape(*shape[:-1], -1), rope).reshape(-1, channels)
        projected = proj.matmul(proj.quantize(attended))
        normed, x_attn = fc1.quantize_layernorm(
            projected, x_flat, self.norm2, input_bias=proj.bias[:channels], return_sum=True
        )
        hidden = fc1.matmul(normed)
        packed = fc2.quantize_gelu(hidden, self.mlp.act.approximate, input_bias=fc1.bias[: fc1.out_features])
        return (x_attn + fc2.matmul(packed, bias=True)).reshape(shape)

    @staticmethod
    def _native(module) -> StaticMXFP8Linear | StaticInt8Linear | None:
        return module if isinstance(module, STATIC_LINEARS) else None

    def _mlp_after_fc1(self, hidden: Tensor, dtype: torch.dtype) -> Tensor:
        hidden = hidden.to(dtype)
        fc2, act = self._native(self.mlp.fc2), self.mlp.act
        if fc2 is not None and isinstance(act, nn.GELU):
            packed = fc2.quantize_gelu(hidden, act.approximate)
            return fc2.forward_prequantized(packed).to(dtype)
        return self.mlp.fc2(act(hidden)).to(dtype)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        if self.fused_epilogues:
            return self._forward_fused(x, rope)
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        qkv = self._native(self.attn.qkv)
        if qkv is not None:
            normed = qkv.quantize_layernorm(x_flat, None, self.norm1)
            qkv_out = qkv.forward_prequantized(normed).to(x.dtype).reshape(*shape[:-1], -1)
            attended = self.attn.proj(self.attn.attend(qkv_out, rope)).reshape(-1, shape[-1])
        else:
            attended = self.attn(self.norm1(x), rope).reshape(-1, shape[-1])
        attended = self.ls1(attended)
        fc1 = self._native(self.mlp.fc1)
        if fc1 is not None and attended.dtype == x_flat.dtype:
            # the norm kernel also writes the residual stream x + attended (same rounding as the BF16 add it replaces)
            normed, x_attn = fc1.quantize_layernorm(attended, x_flat, self.norm2, return_sum=True)
            hidden = self._mlp_after_fc1(fc1.forward_prequantized(normed), x_flat.dtype)
        else:
            x_attn = x_flat + attended
            hidden = self.mlp(self.norm2(x_attn))
        return (x_attn + self.ls2(hidden)).reshape(shape)


def fold_layer_scale(model: nn.Module) -> list[str]:
    """Fold every :class:`TransformerBlock`'s LayerScale into the linear in front of it (``ls1`` into ``attn.proj``,
    ``ls2`` into ``mlp.fc2``: ``gamma * (W x + b) = (gamma W) x + gamma b``) and drop it. Exact in real arithmetic, not
    in floating point; run before quantization (the folded weights are what gets quantized; the calibrated activation
    scales are unaffected: LayerScale sits after these linears)."""
    folded = []
    for name, block in model.named_modules():
        if not is_transformer_block(block):
            continue
        for scale_name, linear in (("ls1", block.attn.proj), ("ls2", block.mlp.fc2)):
            scale = getattr(block, scale_name)
            if isinstance(scale, nn.Identity) or not isinstance(linear, nn.Linear):
                continue
            with torch.no_grad():
                gamma = scale.gamma.to(linear.weight.dtype)
                linear.weight.mul_(gamma[:, None])
                if linear.bias is not None:
                    linear.bias.mul_(gamma.to(linear.bias.dtype))
            setattr(block, scale_name, nn.Identity())
            folded.append(f"{name}.{scale_name}")
    return folded


def enable_fused_epilogues(model: nn.Module) -> list[str]:
    """Switch every eligible :class:`QuantizedTransformerBlock` (MXFP8 linears, LayerScale folded) to its fused
    epilogues; returns the block names."""
    enabled = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizedTransformerBlock) and module.fusable_epilogues():
            module.fused_epilogues = True
            enabled.append(name)
    return enabled
