"""A model FlashIM has never seen, built only from ``flashim.layers``, gets the existing quantization and fusion
passes with no model-specific code: calibrate -> quantize_model -> fused blocks through op dispatch."""

import pytest
import torch
from torch import nn

from flashim import layers, ops
from flashim.quant.apply import quantize_model
from flashim.quant.calibration import Calibrator, Int8Calibrator
from flashim.quant.passes.separable_conv_block import fold_block_batchnorm, fuse_bf16_blocks
from flashim.quant.passes.transformer_block import QuantizedTransformerBlock
from flashim.quant.scheme import INT8_ACTIVATIONS, INT8_WEIGHTS, MXFP8, QuantConfig, QuantScheme


class TinyViT(nn.Module):
    def __init__(self, dim=64, depth=2, heads=2):
        super().__init__()
        self.blocks = nn.ModuleList(layers.TransformerBlock(dim, heads, layer_scale=True) for _ in range(depth))
        self.head = nn.Linear(dim, 8)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TinyConvNet(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*(layers.SeparableConvBlock(channels) for _ in range(2)))

    def forward(self, x):
        return self.body(self.stem(x))


def _vit_targets(model):
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith("blocks.")]


@pytest.mark.parametrize("fmt", ["mxfp8", "int8"])
def test_new_transformer_model_gets_quantized_block_fusion(fmt):
    torch.manual_seed(0)
    model = TinyViT().eval()
    x = torch.randn(2, 10, 64)
    with torch.inference_mode():
        ref = model(x)
        targets = _vit_targets(model)
        calibrator = (Calibrator if fmt == "mxfp8" else Int8Calibrator)(model, targets)
        with calibrator:
            model(x)
        table = calibrator.table()
    weights, acts = (MXFP8, MXFP8) if fmt == "mxfp8" else (INT8_WEIGHTS, INT8_ACTIVATIONS)
    config = QuantConfig({"blocks": QuantScheme(("blocks.*",), weights, acts)})
    report = quantize_model(model, config, table)
    assert report.fused_blocks == ["blocks.0", "blocks.1"]
    assert all(isinstance(b, QuantizedTransformerBlock) for b in model.blocks)
    assert len(report.linears) == 8 and isinstance(model.head, nn.Linear)  # the untargeted head stays dense
    ops.reset_call_counts()
    with torch.inference_mode():
        out = model(x)
    counts = ops.call_counts()
    assert "layernorm_quantize" in counts or "layernorm_quantize_int8" in counts  # fused producer quantization
    assert ((out - ref).norm() / ref.norm()) < 0.1


def test_new_conv_model_gets_separable_block_passes():
    torch.manual_seed(0)
    model = TinyConvNet().eval()
    for block in model.body:
        block.norm.running_var.uniform_(0.5, 1.5)
    x = torch.randn(1, 3, 24, 24)
    with torch.inference_mode():
        ref = model(x)
        assert fold_block_batchnorm(model) == ["body.0", "body.1"]
        assert torch.allclose(model(x), ref, atol=1e-4, rtol=1e-4)  # BN folding is exact
        assert fuse_bf16_blocks(model, ("body.",)) == ["body.0", "body.1"]
        ops.reset_call_counts()
        out = model(x)
    assert ops.call_counts()["depthwise5x5_relu_pointwise"] == {"reference": 2}  # CPU: the op's reference
    assert ((out.float() - ref).norm() / ref.norm()) < 0.02
