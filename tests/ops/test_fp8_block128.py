"""``ops.mxfp8_linear``'s ``cublas-block128`` impl (cuBLAS blockwise FP8, Hopper only; layers with ``activation_block == 128``) against the torch
reference and the default Triton impl, on the FlexGlue linear shapes."""

import pytest
import torch

from flashim import ops
from flashim.quant.formats import mx
from flashim.quant.tensor import QuantizedWeight, quantize_static

pytestmark = pytest.mark.cuda
sm90 = pytest.mark.sm90
NAME = "cublas-block128"
DEV = "cuda"
SHAPES = [(512, 512), (512, 1536), (1024, 512), (1024, 1024)]  # (K, N) of FlexGlue's 81 linears


def _layer(k, n, rows):
    """Activation quantized with the layer's merged (group-of-4 max) exponent, and the flagged weight."""
    torch.manual_seed(0)
    x = (torch.randn(rows, k, device=DEV) * torch.logspace(-1, 1, k, device=DEV)).bfloat16()
    weight = QuantizedWeight.from_float(torch.randn(n, k, device=DEV) * torch.rand(n, 1, device=DEV) * 4)
    exponent = mx.calibrate_block32_exponent(x)
    groups = exponent.reshape(-1, 4)
    exponent = groups.amax(dim=-1, keepdim=True).expand_as(groups).reshape(-1).contiguous()
    flagged = QuantizedWeight(weight.values, weight.scale_exponent, weight.logical_shape, 128)
    return x, exponent, quantize_static(x, exponent), flagged


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@sm90
@pytest.mark.parametrize("rows", [100, 8192])
@pytest.mark.parametrize("k,n", SHAPES)
def test_matches_reference_and_triton(k, n, rows):
    _, _, activation, weight = _layer(k, n, rows)
    bound = ops.mxfp8_linear.bind(weight).warmup()
    assert bound.primary == NAME  # dispatch picks it for a flagged weight, no pin
    ops.reset_call_counts()
    out = bound(activation, None)
    assert ops.call_counts()["mxfp8_linear"] == {NAME: 1}
    reference = ops.mxfp8_linear.get("reference")(activation, None, weight)
    triton = ops.mxfp8_linear.get("triton")(activation, None, weight)
    assert out.shape == (rows, n) and out.dtype == torch.bfloat16
    # BF16 output rounding alone is 1.7e-3; the weight re-encode adds ~6e-4 (measured on real FlexGlue linears)
    assert _rel(out, reference) < 3e-3
    assert _rel(out, triton) < 3e-3


@sm90
def test_unflagged_weights_keep_the_triton_path():
    _, _, activation, weight = _layer(512, 512, 256)
    plain = QuantizedWeight(weight.values, weight.scale_exponent, weight.logical_shape)
    bound = ops.mxfp8_linear.bind(plain).warmup()
    assert bound.primary != NAME and NAME not in bound._prepared
    ops.reset_call_counts()
    bound(activation, None)
    assert NAME not in ops.call_counts()["mxfp8_linear"]


@sm90
@pytest.mark.parametrize("case", ["bias", "rows"])
def test_ineligible_calls_fall_back_to_triton(case):
    _, _, activation, weight = _layer(256, 128, 6 if case == "rows" else 8)
    bias = torch.zeros(128, device=DEV) if case == "bias" else None
    bound = ops.mxfp8_linear.bind(weight).warmup()
    ops.reset_call_counts()
    out = bound(activation, bias)
    assert ops.call_counts()["mxfp8_linear"] == {"triton": 1}
    assert _rel(out, ops.mxfp8_linear.get("reference")(activation, bias, weight)) < 3e-3


@sm90
def test_compiles_and_matches_eager():
    x, exponent, activation, weight = _layer(1024, 1024, 8192)
    bound = ops.mxfp8_linear.bind(weight).warmup()

    def run(value):
        return bound(ops.quantize_mxfp8(value, exponent), None)

    eager = run(x)
    compiled = torch.compile(run, fullgraph=True)(x)
    assert _rel(compiled, eager) < 1e-3
    assert _rel(compiled, ops.mxfp8_linear.get("reference")(activation, None, weight)) < 3e-3


@sm90
def test_cuda_graph_capture_replays_eagerly():
    _, _, activation, weight = _layer(512, 512, 4096)
    bound = ops.mxfp8_linear.bind(weight).warmup()
    expected = bound(activation, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(activation, None)
    graph.replay()
    assert torch.equal(captured, expected)
