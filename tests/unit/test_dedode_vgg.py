"""DeDoDe's VGG encoder stem fusions (``dedode.spec``: ``fused_pool_epilogue``, ``fused_first_conv_epilogue``)."""

import pytest
import torch
from torch import nn

from flashim import ops
from flashim.models.dedode import blocks
from flashim.quant.methods.conv2d import StaticInt8Conv2d


def _int8_vgg(size: str = "19", device: str = "cpu") -> blocks.VGG:
    """A randomly initialised VGG as the int8 preset builds it: BatchNorm folded, the first conv a BF16 ``nn.Conv2d``,
    every other conv a static INT8 one."""
    torch.manual_seed(0)
    vgg = blocks.VGG(size).eval()
    layers = list(vgg.layers)
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.BatchNorm2d):
            layer.running_mean.normal_(0, 0.1), layer.running_var.uniform_(0.5, 2)
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
            layers[i] = nn.utils.fusion.fuse_conv_bn_eval(layer, layers[i + 1])
            layers[i + 1] = nn.Identity()
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Conv2d) and i > 0:
            layers[i] = StaticInt8Conv2d.from_conv(layer, torch.tensor(0.05 * (1 + i % 3)))
    layers[0].bfloat16()
    vgg.layers = nn.ModuleList(layers)
    return vgg.to(device)


def _maps(vgg: blocks.VGG, x: torch.Tensor):
    with torch.no_grad():
        feats, sizes = vgg(x)
    return feats, sizes


def test_pass_patterns():
    vgg = _int8_vgg()
    assert vgg.find_pooled_convs() == {3, 10, 23}  # the last pool (index 39) feeds nothing
    assert vgg.find_first_conv_relu() == 2
    vgg11 = _int8_vgg("11")  # DaD's detector: the first conv feeds a pool, not a quantized conv
    assert vgg11.find_first_conv_relu() is None and 0 not in vgg11.find_pooled_convs()
    assert blocks.VGG("19").find_pooled_convs() == frozenset()  # unquantized layers have no epilogue to extend


def test_last_pool_is_not_run():
    vgg = _int8_vgg()
    x = (torch.randn(1, 3, 48, 64) * 2).bfloat16()
    calls = []
    vgg.layers[-1].register_forward_hook(lambda *_: calls.append(1))
    feats, sizes = _maps(vgg, x)
    assert not calls and [tuple(f.shape[-2:]) for f in feats] == [(48, 64), (24, 32), (12, 16), (6, 8)]
    assert sizes == [f.shape[-2:] for f in feats]


def test_fused_pools_are_bit_identical():
    vgg = _int8_vgg()
    x = (torch.randn(1, 3, 48, 64) * 2).bfloat16()
    want, want_sizes = _maps(vgg, x)
    vgg.pooled_convs = vgg.find_pooled_convs()
    ops.reset_call_counts()
    feats, sizes = _maps(vgg, x)
    assert ops.call_counts()["int8_conv2d_maxpool"] == {"reference": 3}
    assert sizes == want_sizes and all(torch.equal(a, b) for a, b in zip(feats, want))


def test_odd_sizes_keep_the_unfused_semantics():
    vgg = _int8_vgg()
    x = (torch.randn(1, 3, 40, 52) * 2).bfloat16()  # 40, 20, 10, 5: the third conv's output is odd
    want, _ = _maps(vgg, x)
    vgg.pooled_convs = vgg.find_pooled_convs()
    feats, _ = _maps(vgg, x)
    assert all(torch.equal(a, b) for a, b in zip(feats, want))


@pytest.mark.cuda
def test_stem_fusions_are_bit_identical_on_cuda():
    vgg = _int8_vgg(device="cuda")
    x = (torch.randn(1, 3, 96, 128, device="cuda") * 2).bfloat16()
    want, want_sizes = _maps(vgg, x)
    vgg.pooled_convs, vgg.first_conv_relu = vgg.find_pooled_convs(), vgg.find_first_conv_relu()
    ops.reset_call_counts()
    feats, sizes = _maps(vgg, x)
    counts = ops.call_counts()
    assert counts["int8_conv2d_maxpool"] == {"triton": 3} and counts["bias_relu_quantize_int8"] == {"triton": 1}
    assert sizes == want_sizes and all(torch.equal(a, b) for a, b in zip(feats, want))


@pytest.mark.cuda
def test_tensor_core_first_conv_and_patch_embed():
    """The numerics-changing stem passes: the tensor-core first conv feeds layer 3 an INT8 map that is the unfused one
    up to the FP32 summation order (checked at the op level), and ``DinoV2`` takes the bound patch-embedding op."""
    from flashim.platforms import platform_for

    if platform_for("cuda").capability not in ((8, 9), (9, 0)):
        pytest.skip("registered for sm_89 / sm_90")
    vgg = _int8_vgg(device="cuda")
    x = (torch.randn(1, 3, 96, 128, device="cuda") * 2).bfloat16()
    want, _ = _maps(vgg, x)
    relu, conv = vgg.find_first_conv_relu(), vgg.layers[0]
    vgg.tensor_core_first_conv = (relu, ops.conv2d_small_cin_quantize_int8.bind(conv.weight, conv.bias).warmup())
    ops.reset_call_counts()
    feats, _ = _maps(vgg, x)
    assert ops.call_counts()["conv2d_small_cin_quantize_int8"] == {"triton": 1}
    assert "bias_relu_quantize_int8" not in ops.call_counts() or not ops.call_counts()["bias_relu_quantize_int8"]
    for a, b in zip(feats, want):
        assert torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0) > 0.999

    dino = blocks.DinoV2(dim=64, depth=1, heads=2, img_size=56).cuda().bfloat16().eval()
    image = torch.randn(1, 3, 56, 84, device="cuda")
    with torch.no_grad():
        want = dino(image)
        dino.patch_embed_op = ops.patch_embed_tokens.bind(dino.patch_embed.proj.weight, dino.patch_embed.proj.bias).warmup()
        ops.reset_call_counts()
        out = dino(image)
    assert ops.call_counts()["patch_embed_tokens"] == {"cublas": 1} and out.shape == want.shape
    assert torch.nn.functional.cosine_similarity(out.float().flatten(), want.float().flatten(), dim=0) > 0.999
