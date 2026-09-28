import pytest
import torch
from torch import nn

from flashim import ops
from flashim.quant.formats import int8
from flashim.quant.methods.int8_linear import StaticInt8Linear
from flashim.quant.tensor import quantize_int8_static


def _layer(out_f, in_f, device, scale=0.05):
    torch.manual_seed(0)
    linear = nn.Linear(in_f, out_f).to(device)
    return linear, StaticInt8Linear.from_linear(linear, torch.tensor(scale))


def test_weight_quantization_is_per_channel_symmetric():
    torch.manual_seed(0)
    weight = torch.randn(16, 64) * torch.rand(16, 1) * 5
    scale = int8.weight_scales(weight)
    q = int8.quantize_symmetric(weight, scale[:, None])
    assert q.dtype == torch.int8 and q.abs().amax(dim=1).eq(127).all()
    assert ((q.float() * scale[:, None] - weight).abs() <= scale[:, None] / 2 + 1e-6).all()


def test_static_quantization_rounds_half_even_and_saturates():
    x = torch.tensor([[0.5, 1.5, 2.5, -0.5, 1000.0, -1000.0, 0.0]])
    q = quantize_int8_static(x, torch.tensor(1.0))
    assert q.values.tolist() == [[0, 2, 2, 0, 127, -127, 0]]


def test_cpu_close_to_dense():
    x = torch.randn(4, 9, 160) * 3
    linear, packed = _layer(96, 160, "cpu", scale=x.abs().max().item() / 127)
    out = packed(x)
    assert out.shape == (4, 9, 96) and out.dtype == x.dtype
    assert ((out - linear(x)).norm() / linear(x).norm()) < 0.03


def test_conv1x1_matches_linear():
    torch.manual_seed(0)
    conv = nn.Conv2d(64, 32, 1)
    x = torch.randn(2, 5, 64) * 3
    packed = StaticInt8Linear.from_conv1x1(conv, x.abs().max() / 127)
    ref = nn.functional.conv2d(x.reshape(2, 5, 1, 64).permute(0, 3, 1, 2), conv.weight, conv.bias)
    out = packed(x).reshape(2, 5, 32).permute(0, 2, 1)
    assert ((out - ref[..., 0]).norm() / ref.norm()) < 0.03


def test_prequantized_requires_same_scale():
    _, packed = _layer(32, 64, "cpu")
    x = torch.randn(10, 64)
    packed.forward_prequantized(quantize_int8_static(x, packed.input_scale))
    with pytest.raises(ValueError):
        packed.forward_prequantized(quantize_int8_static(x, packed.input_scale * 2))


@pytest.mark.cuda
@pytest.mark.parametrize("rows,k,n", [(5000, 1024, 3072), (300, 1024, 1024), (17, 768, 96), (5, 64, 24)])
def test_gemm_impls_agree(rows, k, n):
    _, layer = _layer(n, k, "cuda")
    q = layer.quantize(torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 2)
    args = (q.matrix(), layer.weight_int8, layer.input_scale, layer.weight_scale, layer.bias)
    ref = ops.int8_linear.get("reference")(*args).float()
    for impl in ("triton", "cublas"):
        out = ops.int8_linear.get(impl)(*args).float()
        assert ((out - ref).abs() <= ref.abs() * 2**-7 + 1e-3).all(), impl


@pytest.mark.cuda
@pytest.mark.parametrize("residual", [False, True])
def test_layernorm_gelu_quantize_match_reference(residual):
    torch.manual_seed(0)
    scale = torch.tensor(0.037, device="cuda")
    x = torch.randn(333, 1000, device="cuda", dtype=torch.bfloat16) * 3
    norm = nn.LayerNorm(1000).cuda()
    norm.weight.data.normal_(), norm.bias.data.normal_()
    r = torch.randn_like(x) if residual else None
    a = ops.layernorm_quantize_int8.get("triton")(x, r, norm.weight, norm.bias, norm.eps, scale)
    b = ops.layernorm_quantize_int8.get("reference")(x, r, norm.weight, norm.bias, norm.eps, scale)
    assert (a.values != b.values).float().mean() < 5e-3
    h = torch.randn(333, 1000, device="cuda", dtype=torch.bfloat16) * 2
    for approximate in ("none", "tanh"):
        a = ops.gelu_quantize_int8.get("triton")(h, scale, approximate)
        b = ops.gelu_quantize_int8.get("reference")(h, scale, approximate)
        assert (a.values != b.values).float().mean() < 5e-3
    q = ops.quantize_int8.get("triton")(h, scale)
    r = ops.quantize_int8.get("reference")(h, scale).values
    # kernels multiply by 1/scale; an exact tie can round the other way (by one quantum)
    assert (q.values.int() - r.int()).abs().max() <= 1 and (q.values != r).float().mean() < 5e-3


@pytest.mark.cuda
@pytest.mark.parametrize("channels,height,width", [(32, 20, 33), (128, 48, 40), (512, 12, 17)])
def test_depthwise_int8_producer_matches_reference(channels, height, width):
    torch.manual_seed(0)
    x = torch.randn(2, channels, height, width, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last
    )
    weight = (torch.randn(25, channels, device="cuda") * 0.2).bfloat16()
    bias = torch.randn(channels, device="cuda")
    scale = torch.tensor(0.03, device="cuda")
    out = ops.depthwise5x5_relu_int8.get("triton")(x, weight, bias, scale)
    ref = ops.depthwise5x5_relu_int8.get("reference")(x, weight, bias, scale)
    assert out.dtype == torch.int8 and out.shape == ref.shape
    assert (out.int() - ref.int()).abs().max() <= 1  # FP32 tap-order differences may flip a rounding
    assert (out != ref).float().mean() < 5e-3


@pytest.mark.cuda
def test_separable_block_and_conv1x1_wrappers():
    from flashim.models.romav2.refiner import RefinerBlock
    from flashim.quant.methods.int8_linear import StaticInt8Conv1x1
    from flashim.quant.passes.separable_conv_block import Int8SeparableConvBlock, fold_block_batchnorm

    torch.manual_seed(0)
    block = RefinerBlock(128).cuda().eval()
    block.norm.running_mean.normal_(), block.norm.running_var.uniform_(0.5, 2)
    x = (
        (torch.randn(2, 128, 24, 24, device="cuda") * 2)
        .bfloat16()
        .contiguous(memory_format=torch.channels_last)
    )
    with torch.no_grad():
        ref = block(x).float()
        fold_block_batchnorm(torch.nn.ModuleList([block]))
        hidden = block.relu(block.conv_depthwise(x.float()))
        fused = Int8SeparableConvBlock(block, hidden.abs().quantile(0.9999) / 127).cuda()
        out = fused(x).float()
    assert out.shape == ref.shape and ((out - ref).norm() / ref.norm()) < 0.05

    conv = torch.nn.Conv2d(64, 96, 1).cuda()
    feats = torch.randn(2, 64, 10, 12, device="cuda", dtype=torch.bfloat16)
    packed = StaticInt8Conv1x1.from_conv(conv, feats.abs().max().float() / 127)
    with torch.no_grad():
        want = torch.nn.functional.conv2d(feats.float(), conv.weight, conv.bias)
    got = packed(feats)
    assert got.shape == want.shape and got.dtype == feats.dtype
    assert ((got.float() - want).norm() / want.norm()) < 0.03


@pytest.mark.cuda
@pytest.mark.parametrize("impl,channels,width", [("tilelang", 128, 40), ("tilelang", 512, 33), ("triton_fused", 128, 45)])
def test_int8_block_kernels_match_reference(impl, channels, width):
    if impl == "tilelang":
        pytest.importorskip("tilelang")
    torch.manual_seed(0)
    x = torch.randn(2, channels, 30, width, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last
    )
    dw = (torch.randn(25, channels, device="cuda") * 0.2).bfloat16()
    dwb = torch.randn(channels, device="cuda")
    scale = torch.tensor(0.03, device="cuda")
    pw = torch.randint(-127, 128, (channels, channels), dtype=torch.int8, device="cuda")
    wscale, pwb = torch.rand(channels, device="cuda") * 0.01, torch.randn(channels, device="cuda").bfloat16()
    if impl == "tilelang":
        got = ops.depthwise5x5_relu_int8.get("tilelang")(x, dw, dwb, scale)
        want = ops.depthwise5x5_relu_int8.get("reference")(x, dw, dwb, scale)
        assert (got.int() - want.int()).abs().max() <= 1
        assert (got != want).float().mean() < 5e-3
    else:
        args = (x, dw, dwb, pw, wscale, pwb, scale)
        got = ops.depthwise5x5_relu_pointwise_int8.get("triton_fused")(*args).float()
        want = ops.depthwise5x5_relu_pointwise_int8.get("reference")(*args).float()
        assert ((got - want).norm() / want.norm()) < 5e-3
