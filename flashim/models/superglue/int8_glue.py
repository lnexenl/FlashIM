"""SuperGlue's GNN layers with static INT8 linears (``int8`` preset) with their glue fused, bit-identical to the
unfused modules' forwards, for any ``(M, N)`` keypoint counts:

- both images stay one ``[M + N, C]`` stream across all 18 layers (no per-layer ``cat`` / ``chunk``);
- one read of ``x`` quantizes it for both packed ``qkv`` and ``mlp.0``'s left half (``ops.quantize_int8_multi``);
- attention output is quantized straight from its SDPA layout (no transpose copy) and ``merge`` writes its quantized
  output directly into ``mlp.0``'s right half (``ops.int8_linear_quantize``, no ``cat``);
- ``mlp.0`` fuses ReLU into its GEMM epilogue (``ops.int8_linear(relu=True)``);
- ``mlp.3`` applies SmoothQuant per-channel scales on input quantization and fuses the residual add into its GEMM
  epilogue (``ops.int8_linear(residual=x)``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.quant.methods.int8_linear import StaticInt8Linear


def _linears(layer) -> tuple[nn.Module, ...]:
    return (layer.attn.qkv, layer.attn.merge, layer.mlp[0], layer.mlp[3])


def unsupported(model: nn.Module) -> str | None:
    """Why ``model`` cannot run the fused INT8 GNN path (``None``: eligible)."""
    for layer in model.gnn.layers:
        if layer.attn.qkv is None:
            return "QKV is not packed (run pack_qkv first)"
        if not all(isinstance(m, StaticInt8Linear) for m in _linears(layer)):
            return "not every GNN linear is static INT8"
        if any(m.channel_scale is not None for m in (layer.attn.qkv, layer.attn.merge, layer.mlp[0])):
            return "SmoothQuant channel scale outside mlp.3"
        if not (isinstance(layer.mlp[1], nn.Identity) and isinstance(layer.mlp[2], nn.ReLU)):
            return "MLP is not folded Linear + ReLU + Linear"
    return None


def probe(model: nn.Module) -> list[str]:
    """Return ops of the fused path lacking a kernel on ``model.device``."""
    device = model.device
    layer = model.gnn.layers[0]
    qkv, merge = layer.attn.qkv, layer.attn.merge
    mlp0, mlp3 = layer.mlp[0], layer.mlp[3]
    c = merge.in_features
    heads = layer.attn.num_heads
    rows = 128
    x = torch.zeros(rows, c, device=device, dtype=torch.bfloat16)
    qx = torch.zeros(rows, c, device=device, dtype=torch.int8)
    wide = torch.zeros(rows, 2 * c, device=device, dtype=torch.int8)
    o = torch.zeros(2, heads, rows // 2, c // heads, device=device, dtype=torch.bfloat16)
    checks = {
        "quantize_int8_multi": ops.quantize_int8_multi.supports(
            x, [qkv.input_scale, mlp0.input_scale], [qx, wide[:, :c]]
        ),
        "int8_linear": ops.int8_linear.supports(qx, qkv.weight_int8, qkv.input_scale, qkv.weight_scale, qkv.bias),
        "quantize_int8_heads": ops.quantize_int8_heads.supports(o, merge.input_scale),
        "int8_linear_quantize": ops.int8_linear_quantize.supports(
            qx, merge.weight_int8, merge.input_scale, merge.weight_scale, merge.bias, mlp0.input_scale, wide[:, c:]
        ),
        "int8_linear(relu)": ops.int8_linear.supports(
            wide, mlp0.weight_int8, mlp0.input_scale, mlp0.weight_scale, mlp0.bias, relu=True
        ),
        "int8_linear(residual)": ops.int8_linear.supports(
            wide, mlp3.weight_int8, mlp3.input_scale, mlp3.weight_scale, mlp3.bias, residual=x
        ),
    }
    return [name for name, ok in checks.items() if not ok]


def _quantize_attn_out(o: Tensor, scale: Tensor) -> Tensor:
    """Quantize SDPA output ``o [B, H, N, D]`` to ``[B * N, H * D]`` INT8 with zero copy in either SDPA layout."""
    if o.is_contiguous():
        return ops.quantize_int8_heads(o, scale).values
    b, h, n, d = o.shape
    flat = o.transpose(1, 2)
    flat = flat.reshape(b * n, h * d) if flat.is_contiguous() else flat.contiguous().reshape(b * n, h * d)
    return ops.quantize_int8(flat, scale).values


def gnn_stream(model: nn.Module, x: Tensor, m: int, n: int) -> Tensor:
    """Run all 18 GNN layers on the concatenated ``[M + N, C]`` BF16 stream with fused INT8 glue."""
    rows, c = x.shape
    qx = torch.empty(rows, c, device=x.device, dtype=torch.int8)
    mlp0_in = torch.empty(rows, 2 * c, device=x.device, dtype=torch.int8)
    for layer, name in zip(model.gnn.layers, model.gnn.names):
        attn = layer.attn
        qkv, merge = attn.qkv, attn.merge
        mlp0, mlp3 = layer.mlp[0], layer.mlp[3]
        heads, head_dim = attn.num_heads, attn.dim
        sdpa = ops.sdpa.get(attn.sdpa_impl)

        ops.quantize_int8_multi(x, [qkv.input_scale, mlp0.input_scale], [qx, mlp0_in[:, :c]])
        qkv_out = ops.int8_linear(qx, qkv.weight_int8, qkv.input_scale, qkv.weight_scale, qkv.bias)
        if m == n:
            q, k, v = qkv_out.view(2, m, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            if name == "cross":
                k, v = k.flip(0), v.flip(0)
            q_o = _quantize_attn_out(sdpa(q, k, v), merge.input_scale)
        else:
            q0, k0, v0 = qkv_out[:m].view(1, m, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            q1, k1, v1 = qkv_out[m:].view(1, n, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            if name == "cross":
                o0, o1 = sdpa(q0, k1, v1), sdpa(q1, k0, v0)
            else:
                o0, o1 = sdpa(q0, k0, v0), sdpa(q1, k1, v1)
            q_o = torch.cat((_quantize_attn_out(o0, merge.input_scale), _quantize_attn_out(o1, merge.input_scale)), 0)

        ops.int8_linear_quantize(
            q_o, merge.weight_int8, merge.input_scale, merge.weight_scale, merge.bias, mlp0.input_scale, mlp0_in[:, c:]
        )
        h = ops.int8_linear(mlp0_in, mlp0.weight_int8, mlp0.input_scale, mlp0.weight_scale, mlp0.bias, relu=True)
        q_h = mlp3.quantize(h)
        x = ops.int8_linear(q_h.values, mlp3.weight_int8, mlp3.input_scale, mlp3.weight_scale, mlp3.bias, residual=x)
    return x
