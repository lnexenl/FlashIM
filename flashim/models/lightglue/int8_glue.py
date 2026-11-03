"""LightGlue's transformer layers with static INT8 linears (``int8`` presets) with their glue fused, bit-identical to
the modules' own forwards (:class:`~flashim.models.lightglue.model.SelfBlock`, ``CrossBlock``, ``ffn_forward``), for a
pair of images with the same number of keypoints:

- both images' residual stream stays one ``[2 N, C]`` tensor (no per-block ``cat`` / ``chunk``); the rotary tables are
  concatenated once per match instead of once per layer; the cross block's FFN runs on both images at once;
- one read of the stream quantizes it for every consumer (``Wqkv`` + ``ffn.0``'s first half; ``to_qk`` + ``to_v`` +
  ``ffn.0``'s first half: ``ops.quantize_int8_multi``);
- ``Wqkv``'s epilogue applies the interleaved RoPE and scatters q / k / v into ``[B, H, N, D]``
  (``ops.lightglue_qkv_rope``); ``to_qk`` / ``to_v`` also write the image-swapped copy the cross attention reads
  (``ops.lightglue_swapped_linear``: no ``flip``);
- the attention output is quantized from its ``[B, H, N, D]`` layout (``ops.quantize_int8_heads``: no transposed copy),
  and ``out_proj`` / ``to_out`` quantize the message straight into ``ffn.0``'s second half (``ops.int8_linear_quantize``:
  no ``cat``);
- LayerNorm -> GELU -> SmoothQuant scale -> INT8 in one kernel (``ops.layernorm_gelu_quantize_int8``), the residual add in
  ``ffn.3``'s epilogue (``ops.int8_linear(residual=)``).

Measured with the prototype of this path on LoMa's int8 matcher (9 layers, 4096 + 4096 keypoints, SageAttention,
CUDA-graph replay of ``features``): L20 loma-l 14.43 -> 12.64 ms, loma-b 7.16 -> 6.11 ms; H20 13.94 -> 11.75 /
7.40 -> 6.15 ms. Not ported from it (they call SageAttention's private kernels): V written in sage's transposed layout
by the epilogues and the key smoothing fused into sage's key quantization, another 0.5 / 0.3 ms on loma-l / loma-b.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.quant.methods.int8_linear import StaticInt8Linear


def _linears(layer) -> tuple[nn.Module, ...]:
    sa, ca = layer.self_attn, layer.cross_attn
    return (sa.Wqkv, sa.out_proj, sa.ffn[0], sa.ffn[3], ca.to_qk, ca.to_v, ca.to_out, ca.ffn[0], ca.ffn[3])


def unsupported(model: nn.Module) -> str | None:
    """Why ``model``'s layers cannot take the fused path (``None``: they can, given the ops have kernels)."""
    if model.config.depth_confidence > 0:
        return "early stopping (depth_confidence > 0)"
    for layer in model.transformers:
        if not all(isinstance(m, StaticInt8Linear) for m in _linears(layer)):
            return "not every transformer linear is static INT8"
        sa, ca = layer.self_attn, layer.cross_attn
        if any(m.channel_scale is not None for m in (sa.Wqkv, sa.out_proj, sa.ffn[0], ca.to_qk, ca.to_v, ca.to_out,
                                                       ca.ffn[0])):
            return "a SmoothQuant channel scale outside ffn.3"
        for block in (sa, ca):
            norm, act = block.ffn[1], block.ffn[2]
            if not (isinstance(norm, nn.LayerNorm) and isinstance(act, nn.GELU) and act.approximate == "none"):
                return "FFN is not LayerNorm + exact GELU"
    return None


def probe(model: nn.Module) -> list[str]:
    """The ops of the fused path that have no kernel on ``model``'s device for its shapes (``Op.supports`` on probe
    tensors of 64 tokens per image; the GELU's libdevice probe compiles a kernel: call before CUDA graph capture)."""
    device = model.device
    layer = model.transformers[0]
    sa, ca = layer.self_attn, layer.cross_attn
    c = sa.out_proj.in_features
    heads = sa.num_heads
    rows, half = 128, 64
    x = torch.zeros(rows, c, device=device, dtype=torch.bfloat16)
    q = torch.zeros(rows, c, device=device, dtype=torch.int8)
    wide = torch.zeros(rows, 2 * c, device=device, dtype=torch.int8)
    table = torch.zeros(2, 1, half, c // heads, device=device, dtype=torch.bfloat16)
    o = torch.zeros(2, heads, half, c // heads, device=device, dtype=torch.bfloat16)
    hidden = torch.zeros(rows, 2 * c, device=device, dtype=torch.bfloat16)
    first, norm, _, last = sa.ffn
    checks = {
        "quantize_int8_multi": ops.quantize_int8_multi.supports(
            x, [ca.to_qk.input_scale, ca.to_v.input_scale, ca.ffn[0].input_scale], [q, q, wide[:, :c]]),
        "lightglue_qkv_rope": ops.lightglue_qkv_rope.supports(
            q, sa.Wqkv.weight_int8, sa.Wqkv.input_scale, sa.Wqkv.weight_scale, sa.Wqkv.bias, table, table, heads),
        "lightglue_swapped_linear": ops.lightglue_swapped_linear.supports(
            q, ca.to_qk.weight_int8, ca.to_qk.input_scale, ca.to_qk.weight_scale, ca.to_qk.bias),
        "quantize_int8_heads": ops.quantize_int8_heads.supports(o, sa.out_proj.input_scale),
        "int8_linear_quantize": ops.int8_linear_quantize.supports(
            q, sa.out_proj.weight_int8, sa.out_proj.input_scale, sa.out_proj.weight_scale, sa.out_proj.bias,
            first.input_scale, wide[:, c:]),
        "layernorm_gelu_quantize_int8": ops.layernorm_gelu_quantize_int8.supports(
            hidden, norm.weight, norm.bias, norm.eps, last.input_scale, channel_scale=last.channel_scale),
        "int8_linear(residual)": ops.int8_linear.supports(
            wide, last.weight_int8, last.input_scale, last.weight_scale, last.bias, residual=x),
    }
    return [name for name, ok in checks.items() if not ok]


def _ffn(ffn: nn.Sequential, f0in: Tensor, x: Tensor) -> Tensor:
    first, norm, _, last = ffn
    hidden = ops.int8_linear(f0in, first.weight_int8, first.input_scale, first.weight_scale, first.bias)
    q = ops.layernorm_gelu_quantize_int8(
        hidden, norm.weight, norm.bias, norm.eps, last.input_scale, channel_scale=last.channel_scale
    )
    return ops.int8_linear(q.values, last.weight_int8, last.input_scale, last.weight_scale, last.bias, residual=x)


def _message(proj: StaticInt8Linear, message: Tensor, first: StaticInt8Linear, f0in: Tensor, c: int) -> None:
    """``proj(message)`` quantized for ``ffn.0`` into the right half of its input (``cat((x, message))``)."""
    q = ops.quantize_int8_heads(message, proj.input_scale)
    ops.int8_linear_quantize(q.values, proj.weight_int8, proj.input_scale, proj.weight_scale, proj.bias,
                             first.input_scale, f0in[:, c:])


def self_block(block, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rows, c = x.shape
    first = block.ffn[0]
    qx = torch.empty(rows, c, device=x.device, dtype=torch.int8)
    f0in = torch.empty(rows, 2 * c, device=x.device, dtype=torch.int8)
    ops.quantize_int8_multi(x, [block.Wqkv.input_scale, first.input_scale], [qx, f0in[:, :c]])
    lin = block.Wqkv
    q, k, v = ops.lightglue_qkv_rope(qx, lin.weight_int8, lin.input_scale, lin.weight_scale, lin.bias, cos, sin,
                                     block.num_heads)
    _message(block.out_proj, ops.sdpa.get(block.sdpa_impl)(q, k, v), first, f0in, c)
    return _ffn(block.ffn, f0in, x)


def cross_block(block, x: Tensor) -> Tensor:
    rows, c = x.shape
    first = block.ffn[0]
    q_qk, q_v = (torch.empty(rows, c, device=x.device, dtype=torch.int8) for _ in range(2))
    f0in = torch.empty(rows, 2 * c, device=x.device, dtype=torch.int8)
    ops.quantize_int8_multi(x, [block.to_qk.input_scale, block.to_v.input_scale, first.input_scale],
                            [q_qk, q_v, f0in[:, :c]])
    qk, qk_swapped = ops.lightglue_swapped_linear(q_qk, block.to_qk.weight_int8, block.to_qk.input_scale,
                                                  block.to_qk.weight_scale, block.to_qk.bias)
    _, v_swapped = ops.lightglue_swapped_linear(q_v, block.to_v.weight_int8, block.to_v.input_scale,
                                                block.to_v.weight_scale, block.to_v.bias, keep=False)

    def heads(t: Tensor) -> Tensor:
        return t.view(2, rows // 2, block.num_heads, -1).transpose(1, 2)

    _message(block.to_out, ops.sdpa.get(block.sdpa_impl)(heads(qk), heads(qk_swapped), heads(v_swapped)), first,
             f0in, c)
    return _ffn(block.ffn, f0in, x)


def features(model: nn.Module, desc0: Tensor, desc1: Tensor, enc0: Tensor, enc1: Tensor) -> tuple[Tensor, Tensor, int]:
    """:meth:`LightGlue.features` for one pair with ``desc0.shape == desc1.shape`` (batch 1), every layer run."""
    x0, x1 = (model.input_proj(d.to(model.compute_dtype, memory_format=torch.contiguous_format)) for d in (desc0, desc1))
    tokens, c = x0.shape[1], x0.shape[2]
    x = torch.cat((x0, x1)).view(2 * tokens, c)
    cos, sin = torch.cat((enc0.to(model.compute_dtype), enc1.to(model.compute_dtype)), dim=1)
    for layer in model.transformers:
        x = self_block(layer.self_attn, x, cos, sin)
        x = cross_block(layer.cross_attn, x)
    model.executed_layers = len(model.transformers)
    x0, x1 = x.view(2, tokens, c).chunk(2)
    return x0, x1, len(model.transformers) - 1
