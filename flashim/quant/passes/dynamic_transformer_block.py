"""Glue fusions for pre-norm transformer blocks whose linears are :class:`DynamicInt8Linear` (per-token dynamic INT8;
DINOv2 in DeDoDe-G's ``int8`` preset), bit-identical to the unfused block.

For ``x + attn(norm1(x))``, ``x + mlp(norm2(x))`` with LayerScale folded away (:func:`fold_layer_scale`):

- ``norm1`` / ``norm2`` and the per-row INT8 quantization of the next linear's input in one kernel
  (``ops.layernorm_quantize_int8_rows``, whose statistics replicate ATen's LayerNorm);
- ``mlp.act`` (exact GELU) in ``fc1``'s GEMM epilogue (``ops.int8_linear(gelu=True)``);
- the residual add in ``proj``'s and ``fc2``'s GEMM epilogues (``ops.int8_linear(residual=...)``).

Each piece is used only where the op has a kernel for the call (``Op.supports``: architecture, dtypes, the libdevice
probe for GELU); otherwise that step runs as the module it replaces would, so the block's output never depends on
which pieces fused. 16 kernels per block become 9 where all of them do.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.layers import TransformerBlock
from flashim.quant.methods.int8_linear import DynamicInt8Linear
from flashim.quant.methods.linear import StaticMXFP8Linear
from flashim.utils import set_submodule


def _dynamic_linears(block: TransformerBlock) -> bool:
    linears = (block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2)
    return (
        all(isinstance(m, DynamicInt8Linear) for m in linears[:2])
        and isinstance(linears[2], (DynamicInt8Linear, StaticMXFP8Linear))
        and isinstance(linears[3], (DynamicInt8Linear, StaticMXFP8Linear))
        and not (isinstance(linears[2], StaticMXFP8Linear) and isinstance(linears[3], StaticMXFP8Linear))
        and all(
            getattr(m, "channel_scale", None) is None and getattr(m, "block_rotate", None) is None
            for m in linears[:3]
        )
    )


def is_fusable(block: nn.Module) -> bool:
    return (
        isinstance(block, TransformerBlock)
        and _dynamic_linears(block)
        and isinstance(block.ls1, nn.Identity)
        and isinstance(block.ls2, nn.Identity)
        and isinstance(block.norm1.weight, Tensor)
        and block.norm1.bias is not None
        and block.norm2.bias is not None
    )


class DynamicInt8TransformerBlock(nn.Module):
    """:class:`~flashim.layers.TransformerBlock` (same submodule names) with the fusions of this module's docstring."""

    fuse_gelu_quantize: bool = False

    def __init__(self, block: TransformerBlock, device: torch.device | None = None) -> None:
        super().__init__()
        if not is_fusable(block):
            raise TypeError("DynamicInt8TransformerBlock needs dynamic INT8 linears and folded LayerScale")
        self.norm1, self.attn, self.norm2, self.mlp = block.norm1, block.attn, block.norm2, block.mlp
        self.ls1, self.ls2 = block.ls1, block.ls2
        self._exact_gelu = isinstance(self.mlp.act, nn.GELU) and self.mlp.act.approximate == "none"
        self._plan: dict[tuple[torch.device, torch.dtype], tuple[bool, bool, bool]] = {}
        if device is not None:
            self.plan(device, self.norm1.weight.dtype)

    def plan(self, device: torch.device, token_dtype: torch.dtype = torch.bfloat16) -> tuple[bool, bool, bool]:
        """``(layernorm+quantize, GELU epilogue, residual epilogue)`` fused for BF16/FP16 tokens on ``device``,
        decided once per ``(device, dtype)`` with probe tensors (the op's ``when`` / libdevice probe may compile a
        kernel: do it before CUDA graph capture, the pass calls this at build time)."""
        key = (device, token_dtype)
        if key not in self._plan:
            channels = self.norm1.weight.shape[0]
            qkv, fc1 = self.attn.qkv, self.mlp.fc1
            rows = torch.zeros(64, channels, device=device, dtype=token_dtype)
            q = torch.zeros(64, channels, device=device, dtype=torch.int8)
            s = torch.ones(64, 1, device=device)
            ln = ops.layernorm_quantize_int8_rows.supports(rows, self.norm1.weight, self.norm1.bias, self.norm1.eps)
            extra = {"out_dtype": token_dtype} if token_dtype == torch.float16 else {}
            gelu = (
                self._exact_gelu
                and isinstance(fc1, DynamicInt8Linear)
                and ops.int8_linear.supports(q, fc1.weight_int8, s, fc1.weight_scale, fc1.bias, gelu=True, **extra)
            )
            residual = ops.int8_linear.supports(
                q, qkv.weight_int8[:channels], s, qkv.weight_scale[:channels], None, residual=rows, **extra)
            self._plan[key] = (ln, gelu, residual)
        return self._plan[key]

    @staticmethod
    def _linear(lin: DynamicInt8Linear, q: Tensor, s: Tensor, **epilogue) -> Tensor:
        return ops.int8_linear(q, lin.weight_int8, s, lin.weight_scale, lin.bias, **epilogue)

    def _norm_quantize(self, norm: nn.LayerNorm, x: Tensor, fused: bool) -> tuple[Tensor, Tensor]:
        if fused:
            return ops.layernorm_quantize_int8_rows(x, norm.weight, norm.bias, norm.eps)
        return ops.quantize_int8_rows(norm(x))

    def _quantize_fc2_input(self, h: Tensor, *, apply_act: bool) -> tuple[Tensor, Tensor]:
        fc2_cs, fc2_rot = self.mlp.fc2.channel_scale, self.mlp.fc2.block_rotate
        if apply_act and fc2_rot is None and fc2_cs is None and self.fuse_gelu_quantize and isinstance(self.mlp.act, nn.GELU):
            return ops.gelu_quantize_int8_rows(h, self.mlp.act.approximate)
        if apply_act:
            h = self.mlp.act(h)
        if fc2_rot is not None:
            return ops.rotate_quantize_int8_rows(h, fc2_rot, channel_scale=fc2_cs)
        if fc2_cs is not None:
            return ops.quantize_int8_rows(h, channel_scale=fc2_cs)
        return ops.quantize_int8_rows(h)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_contiguous():
            return self._unfused(x, rope)
        ln, gelu, residual = self.plan(x.device, x.dtype)
        out_kw = {"out_dtype": x.dtype} if x.dtype == torch.float16 else {}
        batch, tokens, channels = x.shape
        flat = x.reshape(-1, channels)
        attn = self.attn
        q, s = self._norm_quantize(self.norm1, flat, ln)
        qkv = self._linear(attn.qkv, q, s, **out_kw).reshape(batch, tokens, 3, attn.num_heads, channels // attn.num_heads)
        qh, kh, vh = (t.transpose(1, 2) for t in qkv.unbind(2))
        if rope is not None:
            rope_op = ops.rope_qk_2d if getattr(attn, "rope_2d", False) else ops.rope_qk
            qh, kh = rope_op.get(attn.rope_impl)(qh, kh, *rope)
        out = ops.sdpa.get(attn.sdpa_impl)(qh, kh, vh)
        q, s = ops.quantize_int8_rows_heads(out)
        x1 = (
            self._linear(attn.proj, q, s, residual=flat, **out_kw)
            if residual
            else flat + self._linear(attn.proj, q, s, **out_kw)
        )
        if isinstance(self.mlp.fc1, StaticMXFP8Linear):
            h = self.mlp.fc1.matmul(self.mlp.fc1.quantize_layernorm(x1, None, self.norm2), bias=True).to(x1.dtype)
            q, s = self._quantize_fc2_input(h, apply_act=True)
            x2 = (
                self._linear(self.mlp.fc2, q, s, residual=x1, **out_kw)
                if residual
                else x1 + self._linear(self.mlp.fc2, q, s, **out_kw)
            )
            return x2.view(batch, tokens, channels)
        q, s = self._norm_quantize(self.norm2, x1, ln)
        if isinstance(self.mlp.fc2, StaticMXFP8Linear):
            h = self._linear(self.mlp.fc1, q, s, **out_kw)
            if isinstance(self.mlp.act, nn.GELU):
                packed = self.mlp.fc2.quantize_gelu(h, self.mlp.act.approximate)
                x2 = x1 + self.mlp.fc2.matmul(packed, bias=True).to(x1.dtype)
            else:
                x2 = x1 + self.mlp.fc2(self.mlp.act(h)).to(x1.dtype)
            return x2.view(batch, tokens, channels)
        if gelu:
            q, s = self._quantize_fc2_input(self._linear(self.mlp.fc1, q, s, gelu=True, **out_kw), apply_act=False)
        else:
            q, s = self._quantize_fc2_input(self._linear(self.mlp.fc1, q, s, **out_kw), apply_act=True)
        x2 = (
            self._linear(self.mlp.fc2, q, s, residual=x1, **out_kw)
            if residual
            else x1 + self._linear(self.mlp.fc2, q, s, **out_kw)
        )
        return x2.view(batch, tokens, channels)

    def _unfused(self, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


def fuse_dynamic_blocks(model: nn.Module) -> list[str]:
    """Replace every fusable :class:`TransformerBlock` of ``model`` by a :class:`DynamicInt8TransformerBlock`; returns
    their names."""
    fused = []
    for name, module in list(model.named_modules()):
        if is_fusable(module):
            set_submodule(model, name, DynamicInt8TransformerBlock(module, next(module.parameters()).device))
            fused.append(name)
    return fused
