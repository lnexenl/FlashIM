"""The INT8 LightGlue glue ops (``ops.lightglue_*``, ``quantize_int8_multi`` / ``_heads``, ``int8_linear_quantize``,
``layernorm_gelu_quantize_int8``): each Triton impl equals the unfused library path it replaces bit for bit, and tracks
the torch reference."""

import pytest
import torch
import torch.nn.functional as F

from flashim import ops
from flashim.quant.formats.int8 import weight_scales

pytestmark = pytest.mark.cuda
DEV = "cuda"


def _supported(op, *args, **kwargs) -> None:
    if not torch.cuda.is_available() or not op.supports(*args, **kwargs):
        pytest.skip(f"{op.name}: no kernel here")


def _close_codes(a: torch.Tensor, b: torch.Tensor) -> None:
    """INT8 codes of the impl vs the reference (division vs the kernels' reciprocal multiply): at most one step apart,
    on a tiny fraction."""
    diff = (a.int() - b.int()).abs()
    assert diff.max() <= 1 and (diff > 0).float().mean() < 1e-3


def _linear(n: int, k: int, seed: int = 0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, device=DEV, generator=g) * 0.05
    w_scale = weight_scales(w)
    w8 = (w / w_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    bias = (torch.randn(n, device=DEV, generator=g) * 0.1).bfloat16()
    return w8, w_scale.float(), bias, torch.tensor(0.02, device=DEV)


def _int8(rows: int, k: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randint(-127, 128, (rows, k), device=DEV, dtype=torch.int8, generator=g)


@pytest.mark.parametrize("rows,c,outs", [(8192, 512, 3), (1000, 256, 2), (37, 1024, 1)])
def test_quantize_int8_multi(rows, c, outs):
    x = (torch.randn(rows, c, device=DEV) * 3).bfloat16()
    scales = [torch.tensor(s, device=DEV) for s in (0.021, 0.034, 0.05)[:outs]]
    wide = torch.empty(rows, 2 * c, device=DEV, dtype=torch.int8)
    buffers = [wide[:, :c]] + [torch.empty(rows, c, device=DEV, dtype=torch.int8) for _ in range(outs - 1)]
    _supported(ops.quantize_int8_multi, x, scales, buffers)
    ops.quantize_int8_multi.get("triton")(x, scales, buffers)
    for s, out in zip(scales, buffers):
        assert torch.equal(out, ops.quantize_int8.get("triton")(x, s).values)
        _close_codes(out, ops.quantize_int8.get("reference")(x, s).values)


@pytest.mark.parametrize("batch,heads,tokens,dim", [(2, 8, 4096, 64), (2, 4, 1000, 64), (1, 16, 77, 128)])
def test_quantize_int8_heads(batch, heads, tokens, dim):
    o = (torch.randn(batch, heads, tokens, dim, device=DEV) * 2).bfloat16()
    scale = torch.tensor(0.017, device=DEV)
    _supported(ops.quantize_int8_heads, o, scale)
    fast = ops.quantize_int8_heads.get("triton")(o, scale).values
    flat = o.transpose(1, 2).reshape(batch * tokens, heads * dim)
    assert torch.equal(fast, ops.quantize_int8.get("triton")(flat, scale).values)
    _close_codes(fast, ops.quantize_int8_heads.get("reference")(o, scale).values)


@pytest.mark.parametrize("rows,n", [(8192, 512), (8192, 1024), (1000, 2048), (3, 512)])
@pytest.mark.parametrize("smooth", [False, True])
def test_layernorm_gelu_quantize_int8(rows, n, smooth):
    """Equal to ``nn.LayerNorm`` -> ``nn.GELU`` (BF16 modules) -> ``ops.quantize_int8`` with the channel scale."""
    torch.manual_seed(0)
    x = (torch.randn(rows, n, device=DEV) * 4 + torch.randn(n, device=DEV)).bfloat16()
    norm = torch.nn.LayerNorm(n, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(n) + 0.5)
        norm.bias.copy_(torch.randn(n) * 0.1)
    scale = torch.tensor(0.013, device=DEV)
    cs = (torch.rand(n, device=DEV) + 0.5) if smooth else None
    _supported(ops.layernorm_gelu_quantize_int8, x, norm.weight, norm.bias, norm.eps, scale, channel_scale=cs)
    with torch.no_grad():
        fast = ops.layernorm_gelu_quantize_int8.get("triton")(
            x, norm.weight, norm.bias, norm.eps, scale, channel_scale=cs).values
        unfused = ops.quantize_int8.get("triton")(torch.nn.GELU()(norm(x)), scale, channel_scale=cs).values
        ref = ops.layernorm_gelu_quantize_int8.get("reference")(
            x, norm.weight, norm.bias, norm.eps, scale, channel_scale=cs).values
    assert torch.equal(fast, unfused)
    _close_codes(fast, ref)


@pytest.mark.parametrize("rows,k,n", [(8192, 512, 512), (8192, 256, 256), (999, 1024, 1024)])
def test_int8_linear_quantize(rows, k, n):
    a = _int8(rows, k)
    w8, w_scale, bias, a_scale = _linear(n, k)
    out_scale = torch.tensor(0.03, device=DEV)
    wide = torch.zeros(rows, 2 * n, device=DEV, dtype=torch.int8)
    half = wide[:, n:]
    _supported(ops.int8_linear_quantize, a, w8, a_scale, w_scale, bias, out_scale, half)
    ops.int8_linear_quantize.get("triton")(a, w8, a_scale, w_scale, bias, out_scale, half)
    y = ops.int8_linear.get("triton")(a, w8, a_scale, w_scale, bias)
    assert torch.equal(half, ops.quantize_int8.get("triton")(y, out_scale).values)
    assert not wide[:, :n].any()  # the other half untouched
    ref = torch.empty_like(half)
    ops.int8_linear_quantize.get("reference")(a, w8, a_scale, w_scale, bias, out_scale, ref)
    _close_codes(half, ref)


@pytest.mark.parametrize("tokens,heads,dim", [(4096, 4, 64), (4096, 8, 64), (1000, 16, 64), (77, 2, 128)])
def test_lightglue_qkv_rope(tokens, heads, dim):
    """Equal to ``int8_linear`` -> ``view / permute`` -> ``rope_qk_interleaved`` (SelfBlock.attend's path)."""
    c = heads * dim
    a = _int8(2 * tokens, c)
    w8, w_scale, bias, a_scale = _linear(3 * c, c)
    angles = torch.rand(2, 1, tokens, dim // 2, device=DEV) * 6
    cos, sin = (f(angles).repeat_interleave(2, dim=-1).bfloat16() for f in (torch.cos, torch.sin))
    _supported(ops.lightglue_qkv_rope, a, w8, a_scale, w_scale, bias, cos, sin, heads)
    fast = ops.lightglue_qkv_rope.get("triton")(a, w8, a_scale, w_scale, bias, cos, sin, heads)
    y = ops.int8_linear.get("triton")(a, w8, a_scale, w_scale, bias)
    q, k, v = y.view(2, tokens, 3, heads, dim).permute(2, 0, 3, 1, 4)
    q, k = ops.rope_qk_interleaved(q, k, cos, sin)
    assert fast.shape == (3, 2, heads, tokens, dim)
    assert torch.equal(fast[0], q) and torch.equal(fast[1], k) and torch.equal(fast[2], v)
    ref = ops.lightglue_qkv_rope.get("reference")(a, w8, a_scale, w_scale, bias, cos, sin, heads)
    assert (fast.float() - ref.float()).abs().max() <= 2 ** -6 * ref.float().abs().max()


@pytest.mark.parametrize("keep", [True, False])
@pytest.mark.parametrize("tokens,c", [(4096, 512), (1000, 256)])
def test_lightglue_swapped_linear(tokens, c, keep):
    a = _int8(2 * tokens, c)
    w8, w_scale, bias, a_scale = _linear(c, c)
    _supported(ops.lightglue_swapped_linear, a, w8, a_scale, w_scale, bias, keep=keep)
    y, swapped = ops.lightglue_swapped_linear.get("triton")(a, w8, a_scale, w_scale, bias, keep=keep)
    expected = ops.int8_linear.get("triton")(a, w8, a_scale, w_scale, bias)
    assert (y is None) == (not keep) and (y is None or torch.equal(y, expected))
    assert torch.equal(swapped, expected.view(2, tokens, c).flip(0).reshape(2 * tokens, c))
    _, ref_swapped = ops.lightglue_swapped_linear.get("reference")(a, w8, a_scale, w_scale, bias, keep=True)
    assert F.cosine_similarity(swapped.float().flatten(), ref_swapped.float().flatten(), dim=0) > 0.9999
