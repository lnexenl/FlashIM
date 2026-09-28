"""Fusion pass for post-norm deformable-attention encoders whose linears are static MXFP8.

Every quantize step is fused into the kernel that produces its input, and GEMM biases are folded into the
next elementwise kernel (LayerNorm / GELU), so no dense activation is quantized in a separate pass::

    src ─quantize→ value_proj (+bias in the GEMM epilogue) → value
    src+pos ─ attn.project_query → offsets, logits ──► deformable attention (softmax fused) ─quantize→ output_proj
    LayerNorm(src + output_proj + bias) ──► dense h and quantize(h) ─→ fused MLP: linear1, + bias, GELU,
        quantize and linear2 in one kernel (the hidden activation stays on chip)
    LayerNorm(h + linear2 + bias) ──► dense src and quantize(src) for the next layer's value_proj
"""

from __future__ import annotations

from torch import Tensor, nn

from flashim import ops
from flashim.quant.methods.linear import StaticMXFP8Linear

_MAIN_LINEARS = (("self_attn", "value_proj"), ("self_attn", "output_proj"), ("linear1",), ("linear2",))


def _resolve(layer: nn.Module, path: tuple[str, ...]):
    for part in path:
        layer = getattr(layer, part, None)
        if layer is None:
            return None
    return layer


def is_deformable_encoder(module: nn.Module) -> bool:
    """Structural match: ``layers`` of post-norm deformable encoder layers with packed value/output/FFN linears."""
    layers = getattr(module, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        return False
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not all(hasattr(attn, n) for n in ("project_query", "n_heads", "n_levels", "n_points")):
            return False
        if not all(hasattr(layer, n) for n in ("norm1", "norm2")):
            return False
        if not all(isinstance(_resolve(layer, path), StaticMXFP8Linear) for path in _MAIN_LINEARS):
            return False
    return True


class QuantizedDeformableEncoder(nn.Module):
    """Runs an encoder's (already packed) layers with fused producer -> quantize kernels.

    Shares the original ``layers`` (same module paths, buffers and calibration).
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.layers = encoder.layers
        self.num_layers = len(self.layers)

    def forward(self, src: Tensor, pos: Tensor, reference: Tensor, shapes: tuple[tuple[int, int], ...]) -> Tensor:
        n, length, _ = src.shape
        src_q = ops.quantize_mxfp8(src, self.layers[0].self_attn.value_proj.input_exponent)
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            value = attn.value_proj.matmul(src_q, bias=True).view(n, length, attn.n_heads, -1)
            offsets, logits = attn.project_query(src + pos)
            out = ops.deformable_attention(value, offsets, logits, reference, shapes, softmax=True)
            out = attn.output_proj.matmul(ops.quantize_mxfp8(out, attn.output_proj.input_exponent))
            h_q, h = ops.layernorm_quantize(
                out, src, layer.norm1.weight, layer.norm1.bias, layer.norm1.eps,
                layer.linear1.input_exponent, input_bias=attn.output_proj.bias, return_dense=True,
            )
            ffn = ops.mxfp8_gelu_mlp(
                h_q, layer.linear1.packed_weight(), layer.linear1.bias, layer.linear2.input_exponent,
                layer.linear2.packed_weight(),
            )[:, : layer.linear2.out_features].reshape(n, length, -1)
            if index + 1 < self.num_layers:
                src_q, src = ops.layernorm_quantize(
                    ffn, h, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps,
                    self.layers[index + 1].self_attn.value_proj.input_exponent,
                    input_bias=layer.linear2.bias, return_dense=True,
                )
            else:
                src = layer.norm2(h + (ffn + layer.linear2.bias_bf16))
        return src
