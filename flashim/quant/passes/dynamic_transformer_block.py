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
from flashim.utils import set_submodule


def _dynamic_linears(block: TransformerBlock) -> bool:
    return all(isinstance(m, DynamicInt8Linear) for m in (block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2))


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
    """:class:`~flashim.layers.TransformerBlock` (same submodule names) with the fusions of this module's docstring.
    ``rope`` is not supported (DINOv2 has none): such a call runs the original block."""

    def __init__(self, block: TransformerBlock, device: torch.device | None = None) -> None:
        super().__init__()
        if not is_fusable(block):
            raise TypeError("DynamicInt8TransformerBlock needs dynamic INT8 linears and folded LayerScale")
        self.norm1, self.attn, self.norm2, self.mlp = block.norm1, block.attn, block.norm2, block.mlp
        self.ls1, self.ls2 = block.ls1, block.ls2
        self._exact_gelu = isinstance(self.mlp.act, nn.GELU) and self.mlp.act.approximate == "none"
        self._plan: dict[torch.device, tuple[bool, bool, bool]] = {}
        if device is not None:
            self.plan(device)

    def plan(self, device: torch.device) -> tuple[bool, bool, bool]:
        """``(layernorm+quantize, GELU epilogue, residual epilogue)`` fused for BF16 tokens on ``device``, decided once
        per device with probe tensors (the op's ``when`` / libdevice probe may compile a kernel: do it before CUDA graph
        capture, the pass calls this at build time)."""
        if device not in self._plan:
            channels = self.norm1.weight.shape[0]
            qkv, fc1 = self.attn.qkv, self.mlp.fc1
            rows = torch.zeros(64, channels, device=device, dtype=torch.bfloat16)
            q = torch.zeros(64, channels, device=device, dtype=torch.int8)
            s = torch.ones(64, 1, device=device)
            ln = ops.layernorm_quantize_int8_rows.supports(rows, self.norm1.weight, self.norm1.bias, self.norm1.eps)
            gelu = self._exact_gelu and ops.int8_linear.supports(
                q, fc1.weight_int8, s, fc1.weight_scale, fc1.bias, gelu=True)
            residual = ops.int8_linear.supports(
                q, qkv.weight_int8[:channels], s, qkv.weight_scale[:channels], None, residual=rows)
            self._plan[device] = (ln, gelu, residual)
        return self._plan[device]

    @staticmethod
    def _linear(lin: DynamicInt8Linear, q: Tensor, s: Tensor, **epilogue) -> Tensor:
        return ops.int8_linear(q, lin.weight_int8, s, lin.weight_scale, lin.bias, **epilogue)

    def _norm_quantize(self, norm: nn.LayerNorm, x: Tensor, fused: bool) -> tuple[Tensor, Tensor]:
        if fused:
            return ops.layernorm_quantize_int8_rows(x, norm.weight, norm.bias, norm.eps)
        return ops.quantize_int8_rows(norm(x))

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        if rope is not None or x.dtype != torch.bfloat16 or not x.is_contiguous():
            return self._unfused(x, rope)
        ln, gelu, residual = self.plan(x.device)
        batch, tokens, channels = x.shape
        flat = x.reshape(-1, channels)
        attn = self.attn
        q, s = self._norm_quantize(self.norm1, flat, ln)
        qkv = self._linear(attn.qkv, q, s).reshape(batch, tokens, 3, attn.num_heads, channels // attn.num_heads)
        qh, kh, vh = (t.transpose(1, 2) for t in qkv.unbind(2))
        out = ops.sdpa.get(attn.sdpa_impl)(qh, kh, vh).transpose(1, 2).reshape(-1, channels)
        q, s = ops.quantize_int8_rows(out if out.is_contiguous() else out.contiguous())
        x1 = self._linear(attn.proj, q, s, residual=flat) if residual else flat + self._linear(attn.proj, q, s)
        q, s = self._norm_quantize(self.norm2, x1, ln)
        if gelu:
            q, s = ops.quantize_int8_rows(self._linear(self.mlp.fc1, q, s, gelu=True))
        else:
            q, s = ops.quantize_int8_rows(self.mlp.act(self._linear(self.mlp.fc1, q, s)))
        x2 = self._linear(self.mlp.fc2, q, s, residual=x1) if residual else x1 + self._linear(self.mlp.fc2, q, s)
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
