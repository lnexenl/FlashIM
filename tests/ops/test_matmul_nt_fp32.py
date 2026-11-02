"""``ops.matmul_nt_fp32``: the cuBLAS impl (BF16 / FP16 tensor cores, FP32 accumulation and output) against the
torch reference."""

import pytest
import torch

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


# tiny, odd (unaligned K / rows: cuBLAS' unaligned kernels), tall / wide, and LoMa's similarity shapes (dim 256 / 1024)
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (17, 33, 65), (129, 7, 255), (1000, 1023, 31),
                                   (2047, 4093, 256), (4096, 4096, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [1.0, 512**-0.5])
def test_matches_reference(shape, dtype, scale):
    torch.manual_seed(0)
    m, n, k = shape
    a, b = torch.randn(m, k, device=DEV).to(dtype), torch.randn(n, k, device=DEV).to(dtype)
    assert ops.matmul_nt_fp32.explain(a, b, scale)["cublas"] == "selected"
    out = ops.matmul_nt_fp32.get("cublas")(a, b, scale)
    ref = ops.matmul_nt_fp32.get("reference")(a, b, scale)
    assert out.dtype == torch.float32 and out.shape == (m, n)
    # exact products in FP32, sums in another order: the FP32 rounding bound of a length-k dot product
    bound = k * 2**-23 * (a.abs().double() @ b.abs().double().t()) * scale
    assert ((out.double() - ref.double()).abs() <= bound + 1e-30).all()
    exact = (a.double() @ b.double().t()) * scale
    assert (out.double() - exact).abs().max() <= 1e-5 * exact.abs().max() + 1e-30


def test_fp32_inputs_fall_back_to_the_reference():
    a, b = torch.randn(8, 16, device=DEV), torch.randn(4, 16, device=DEV)
    assert ops.matmul_nt_fp32.explain(a, b, 0.5)["reference"] == "selected"
    torch.testing.assert_close(ops.matmul_nt_fp32(a, b, 0.5), (a @ b.t()) * 0.5)


def test_captures_in_a_cuda_graph_and_compiles():
    a, b = torch.randn(300, 64, device=DEV).bfloat16(), torch.randn(200, 64, device=DEV).bfloat16()
    ref = ops.matmul_nt_fp32(a, b, 0.125)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ops.matmul_nt_fp32(a, b, 0.125)
    graph.replay()
    assert torch.equal(out, ref)
    compiled = torch.compile(lambda x, y: ops.matmul_nt_fp32(x, y, 0.125), fullgraph=True)
    torch.testing.assert_close(compiled(a, b), ref, atol=1e-5, rtol=1e-5)
