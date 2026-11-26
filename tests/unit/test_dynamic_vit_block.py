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


def test_rotation_matrices_properties():
    from flashim.quant.rotation import hadamard, layernorm_orthogonal, signed_hadamard

    for n in (16, 32, 64, 128):
        h = hadamard(n, dtype=torch.float64)
        sh = signed_hadamard(n, seed=0, dtype=torch.float64)
        torch.testing.assert_close(h @ h.T, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(sh @ sh.T, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=0)
    for n in (48, 64, 768, 1024):
        q = layernorm_orthogonal(n, dtype=torch.float64)
        torch.testing.assert_close(q @ q.T, torch.eye(n, dtype=torch.float64), atol=1e-11, rtol=0)
        ones = torch.ones(n, dtype=torch.float64)
        torch.testing.assert_close(q @ ones, ones, atol=1e-11, rtol=0)
        x = torch.randn(4, n, dtype=torch.float64) * 5 + 3
        ln_rot = torch.nn.functional.layer_norm(x @ q, (n,), eps=1e-6)
        ln_ref = torch.nn.functional.layer_norm(x, (n,), eps=1e-6) @ q
        torch.testing.assert_close(ln_rot, ln_ref, atol=1e-10, rtol=1e-10)


def test_fused_block_with_block_rotate_and_hybrid_mxfp8_cpu():
    from flashim.quant.formats import mx
    from flashim.quant.methods.linear import StaticMXFP8Linear
    from flashim.quant.rotation import signed_hadamard

    torch.manual_seed(0)
    x = torch.randn(2, 13, 256, dtype=torch.bfloat16) * 2
    # 1) DynamicInt8 block with block_rotate on fc2 + fuse_gelu_quantize
    blk = TransformerBlock(256, 4, eps=1e-6, layer_scale=False)
    for owner, name in ((blk.attn, "qkv"), (blk.attn, "proj"), (blk.mlp, "fc1")):
        setattr(owner, name, DynamicInt8Linear.from_linear(getattr(owner, name)))
    blk.mlp.fc2 = DynamicInt8Linear.from_linear(blk.mlp.fc2, block_rotate=signed_hadamard(64))
    blk.attn.sdpa_impl = None
    blk = blk.to(torch.bfloat16).eval()
    with torch.inference_mode():
        want = blk(x)
        fused = DynamicInt8TransformerBlock(blk, torch.device("cpu"))
        fused.fuse_gelu_quantize = True
        assert torch.equal(fused(x), want)

    # 2) Hybrid DynamicInt8 + StaticMXFP8 on fc1 or fc2
    for mxfp8_attr in ("fc1", "fc2"):
        hblk = TransformerBlock(256, 4, eps=1e-6, layer_scale=False)
        for owner, name in ((hblk.attn, "qkv"), (hblk.attn, "proj")):
            setattr(owner, name, DynamicInt8Linear.from_linear(getattr(owner, name)))
        other = "fc2" if mxfp8_attr == "fc1" else "fc1"
        lin_m = getattr(hblk.mlp, mxfp8_attr)
        exp = mx.calibrate_block32_exponent(torch.randn(8, lin_m.in_features))
        setattr(hblk.mlp, mxfp8_attr, StaticMXFP8Linear.from_linear(lin_m, exp))
        setattr(
            hblk.mlp,
            other,
            DynamicInt8Linear.from_linear(
                getattr(hblk.mlp, other), block_rotate=signed_hadamard(64) if other == "fc2" else None
            ),
        )
        hblk.attn.sdpa_impl = None
        hblk = hblk.to(torch.bfloat16).eval()
        assert is_fusable(hblk)
        with torch.inference_mode():
            hfused = DynamicInt8TransformerBlock(hblk, torch.device("cpu"))
            hfused.fuse_gelu_quantize = True
            out_f, out_u = hfused(x), hblk(x)
            assert out_f.shape == out_u.shape and out_f.dtype == x.dtype
            assert ((out_f.float() - out_u.float()).norm() / out_u.float().norm()) < 0.02
