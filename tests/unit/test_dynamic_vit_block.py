"""``DynamicInt8TransformerBlock``: the fused block equals the unfused ``TransformerBlock`` it replaces."""

import pytest
import torch
from torch import nn

from flashim.layers import TransformerBlock
from flashim.quant.methods.int8_linear import DynamicInt8Linear
from flashim.quant.passes.dynamic_transformer_block import (
    DynamicInt8TransformerBlock,
    fuse_dynamic_blocks,
    is_fusable,
)


def _block(dtype, device):
    torch.manual_seed(0)
    block = TransformerBlock(256, 4, eps=1e-6, layer_scale=False)
    for module in block.modules():
        if isinstance(module, nn.LayerNorm):
            module.weight.data.normal_()
            module.bias.data.normal_()
    for owner, name in ((block.attn, "qkv"), (block.attn, "proj"), (block.mlp, "fc1"), (block.mlp, "fc2")):
        setattr(owner, name, DynamicInt8Linear.from_linear(getattr(owner, name)))
    block.attn.sdpa_impl = None
    return block.to(device, dtype).eval()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_fused_block_equals_unfused(device):
    dtype = torch.bfloat16
    block = _block(dtype, device)
    x = torch.randn(2, 37, 256, device=device, dtype=dtype) * 2
    x[0, 3] *= 100
    with torch.inference_mode():
        want = block(x)
        fused = DynamicInt8TransformerBlock(block, torch.device(device))
        assert torch.equal(fused(x), want)
        assert torch.equal(fused._unfused(x, None), want)


def test_fuse_dynamic_blocks_replaces_only_fusable_blocks():
    model = nn.ModuleDict({
        "a": _block(torch.bfloat16, "cpu"),
        "b": TransformerBlock(256, 4, layer_scale=True),  # float linears, LayerScale
    })
    assert is_fusable(model["a"]) and not is_fusable(model["b"])
    assert fuse_dynamic_blocks(model) == ["a"]
    assert isinstance(model["a"], DynamicInt8TransformerBlock) and isinstance(model["b"], TransformerBlock)
