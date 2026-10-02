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


def test_indexed_separable_block_keeps_sequential_names_and_gets_the_passes():
    """DeDoDe / RIPE style ``nn.Sequential(conv, bn, relu, conv)`` blocks: upstream state-dict keys, same layer type."""
    torch.manual_seed(0)
    channels = 32
    upstream = nn.Sequential(
        nn.Conv2d(channels, channels, 5, 1, 2, groups=channels), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 1),
    ).eval()
    upstream[1].running_var.uniform_(0.5, 1.5)
    block = layers.IndexedSeparableConvBlock(channels).eval()
    assert list(block.state_dict()) == list(upstream.state_dict())
    block.load_state_dict(upstream.state_dict(), strict=True)
    assert block.conv_depthwise is block.get_submodule("0") and block.conv_pointwise is block.get_submodule("3")
    assert block.pointwise_name == "3" and not block.fuse_quantized
    x = torch.randn(1, channels, 20, 28)
    with torch.inference_mode():
        ref = upstream(x.clone())
        assert torch.equal(block(x.clone()), ref)
        model = nn.Sequential(block, layers.IndexedSeparableConvBlock(48).eval())  # 48: no fused kernel
        assert fold_block_batchnorm(model) == ["0", "1"]
        assert isinstance(block.norm, nn.Identity) and "1" in dict(block.named_children())
        assert fuse_bf16_blocks(model, ("",), channels=(16, 32, 64, 128)) == ["0"]  # only the supported channel count
    assert isinstance(model[1], layers.IndexedSeparableConvBlock)


def test_indexed_separable_block_pointwise_is_quantized_like_a_plain_conv():
    """``fuse_quantized`` off: an INT8 scheme on ``<block>.3`` packs that conv; the block itself is not replaced."""
    from flashim.quant.methods.int8_linear import StaticInt8Conv1x1

    torch.manual_seed(0)
    model = nn.Sequential(layers.IndexedSeparableConvBlock(32)).eval()
    fold_block_batchnorm(model)
    config = QuantConfig({"conv": QuantScheme(("0.3",), INT8_WEIGHTS, INT8_ACTIVATIONS)}, name="int8")
    x = torch.randn(1, 32, 12, 12)
    with torch.inference_mode(), Int8Calibrator(model, ["0.3"]) as calibrator:
        model(x)
    report = quantize_model(model, config, calibrator.table())
    assert isinstance(model[0], layers.IndexedSeparableConvBlock) and isinstance(model[0].conv_pointwise, StaticInt8Conv1x1)
    assert report.fused_conv_blocks == []


def test_blocks_fused_before_quantization_leave_the_quantized_set():
    """A separable block replaced by the BF16 fused block before ``quantize_model`` no longer has its pointwise conv:
    the scheme's target stops matching (unused calibration entries are fine), other targets are still packed."""
    from flashim.quant.methods.int8_linear import StaticInt8Conv1x1
    from flashim.quant.passes.separable_conv_block import BF16SeparableConvBlock

    torch.manual_seed(0)
    model = nn.Sequential(layers.IndexedSeparableConvBlock(32), layers.IndexedSeparableConvBlock(48)).eval()
    model[1] = nn.Sequential(nn.Conv2d(32, 48, 1), model[1])  # 48 channels: no fused kernel, stays a plain block
    fold_block_batchnorm(model)
    targets = ["0.3", "1.1.3"]
    x = torch.randn(1, 32, 12, 12)
    with torch.inference_mode(), Int8Calibrator(model, targets) as calibrator:
        model(x)
    table = calibrator.table()
    assert fuse_bf16_blocks(model, ("",), channels=(16, 32, 64, 128)) == ["0"]
    config = QuantConfig({"conv": QuantScheme(("0.3", "1.1.3"), INT8_WEIGHTS, INT8_ACTIVATIONS)}, name="int8")
    quantize_model(model, config, table)
    assert isinstance(model[0], BF16SeparableConvBlock)
    assert isinstance(model[1][1], layers.IndexedSeparableConvBlock)
    assert isinstance(model[1][1].conv_pointwise, StaticInt8Conv1x1)
