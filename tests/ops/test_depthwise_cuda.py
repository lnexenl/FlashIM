"""CUDA C++ depthwise 5x5 kernels (``backends/cuda/csrc/depthwise.cu``): bit-identical to the Triton kernels they replace
(same FMA order, ``div.full.f32`` reciprocal, ``mma.sync`` chain started from the bias), checked against the torch
reference for the usual FP32 tap-order differences. ``get("cuda")`` pins the impl, so it runs on any architecture with
nvcc; dispatch only selects it on sm_89 / sm_90, the architectures it was measured on."""

import pytest
import torch

from flashim import ops
from flashim.backends.cuda import conv as cuda_conv
from flashim.platforms import platform_for

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.fixture(autouse=True)
def _needs_nvcc():
    if not cuda_conv._depthwise_available():
        pytest.skip("nvcc unavailable")


def _selected_on_measured_archs() -> bool:
    return platform_for(DEV).capability in ((8, 9), (9, 0))


def _producer_inputs(batch, channels, height, width, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(batch, channels, height, width, device=DEV) * 2).bfloat16().contiguous(memory_format=torch.channels_last)
    return x, (torch.randn(25, channels, device=DEV) * 0.2).bfloat16(), torch.randn(channels, device=DEV) * 0.1, torch.tensor(0.03, device=DEV)


# DeDoDe's decoder shapes at 1200 px (dedode: 512 / 256 channels; dedode-g adds 768, odd widths) and ragged ones: widths
# 1 and 7, a width above the tile, a batch of 2 / 3, channel counts outside the template list (192, 384, 1024).
PRODUCER_SHAPES = [
    (1, 512, 112, 150), (1, 256, 224, 294), (1, 768, 64, 84), (1, 512, 112, 147), (1, 64, 5, 7), (2, 128, 33, 17),
    (1, 256, 37, 53), (2, 512, 20, 61), (1, 64, 1, 1), (1, 128, 3, 200), (1, 384, 11, 29), (2, 192, 30, 31),
    (1, 1024, 8, 9), (3, 256, 17, 147),
]


@pytest.mark.parametrize("shape", PRODUCER_SHAPES)
def test_depthwise_int8_producer_cuda_is_bit_identical_to_triton(shape):
    x, w, b, s = _producer_inputs(*shape)
    got = ops.depthwise5x5_relu_int8.get("cuda")(x, w, b, s)
    assert torch.equal(got, ops.depthwise5x5_relu_int8.get("triton")(x, w, b, s))
    want = ops.depthwise5x5_relu_int8.get("reference")(x, w, b, s)
    assert got.dtype == torch.int8 and got.shape == want.shape
    assert (got.int() - want.int()).abs().max() <= 1 and (got != want).float().mean() < 5e-3  # FP32 tap order only


@pytest.mark.parametrize("tile", [(4, 8, y) for y in (1, 2, 4, 8)])
def test_depthwise_int8_producer_cuda_tiles_agree(tile):
    """Every column-group count the sweeps pick from computes the same bits, ragged edges included."""
    for shape in ((1, 256, 37, 53), (2, 64, 20, 9)):
        x, w, b, s = _producer_inputs(*shape)
        got = cuda_conv.launch_depthwise5x5_relu_int8(x, w, b, s.reshape(1), tile)
        assert torch.equal(got, ops.depthwise5x5_relu_int8.get("triton")(x, w, b, s))


def test_depthwise_int8_producer_cuda_reciprocal_ties():
    """``a * (1 / scale)`` with the hardware reciprocal rounds exact .5 ties as the Triton kernel does (IEEE ``a / scale``
    would give 113 for 3.375 / 0.03, the kernels 112)."""
    x, w, b, _ = _producer_inputs(1, 64, 9, 9)
    scale = torch.tensor(0.03, device=DEV)
    w.zero_(), b.fill_(3.375)  # every output is relu(3.375) = 3.375
    got = ops.depthwise5x5_relu_int8.get("cuda")(x, w, b, scale)
    assert torch.equal(got, ops.depthwise5x5_relu_int8.get("triton")(x, w, b, scale))


def test_depthwise_int8_producer_cuda_dispatch_and_views():
    x, w, b, s = _producer_inputs(1, 256, 41, 45)
    assert (ops.depthwise5x5_relu_int8.explain(x, w, b, s)["cuda"] == "selected") is _selected_on_measured_archs()
    want = ops.depthwise5x5_relu_int8.get("triton")(x, w, b, s)
    # a view at an odd element offset (bf16x2 loads need 4-byte alignment) is copied by the wrapper
    padded = torch.zeros(1, 41, 45, 257, device=DEV, dtype=torch.bfloat16)
    padded[..., 1:] = x.permute(0, 2, 3, 1)
    view = padded[..., 1:].permute(0, 3, 1, 2)
    assert view.data_ptr() % 4 and not view.is_contiguous(memory_format=torch.channels_last)  # not eligible as is
    assert torch.equal(ops.depthwise5x5_relu_int8.get("cuda")(view.contiguous(memory_format=torch.channels_last), w, b, s), want)
    odd = torch.zeros(1, 41, 45, 257, device=DEV, dtype=torch.bfloat16)
    odd.view(-1)[1 : 1 + x.numel()] = x.permute(0, 2, 3, 1).reshape(-1)
    flat = odd.view(-1)[1 : 1 + x.numel()].view(1, 41, 45, 256).permute(0, 3, 1, 2)  # channels-last, 2-byte aligned only
    assert flat.data_ptr() % 4 and flat.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(ops.depthwise5x5_relu_int8.get("cuda")(flat, w, b, s), want)


@pytest.mark.parametrize("channels", [32, 64, 128])
@pytest.mark.parametrize(
    "batch,hw",
    [(1, (5, 7)), (2, (33, 17)), (1, (37, 53)), (2, (20, 61)), (1, (64, 64)), (1, (3, 130)), (1, (1, 1)), (3, (17, 147)),
     (1, (448, 588)), (1, (112, 150))],
)
def test_depthwise_relu_pointwise_cuda_is_bit_identical_to_triton(channels, batch, hw):
    """The identity depends on Triton lowering ``tl.dot(a, b) + bias`` to ``tt.dot(a, b, acc=bias)``: an ``mma.sync`` chain
    started from the bias. A Triton upgrade that changes that shows up here, not as drift."""
    torch.manual_seed(2)
    x = (torch.randn(batch, channels, *hw, device=DEV) * 2).bfloat16().contiguous(memory_format=torch.channels_last)
    dw = (torch.randn(25, channels, device=DEV) * 0.2).bfloat16()
    dwb = (torch.randn(channels, device=DEV) * 0.1).bfloat16()
    pw = (torch.randn(channels, channels, device=DEV) * 0.2).bfloat16().contiguous()
    pwb = torch.randn(channels, device=DEV).bfloat16()
    got = ops.depthwise5x5_relu_pointwise.get("cuda")(x, dw, dwb, pw, pwb)
    for tile in ((4, 4, 4), (4, 8, 4), (4, 8, 8)):  # the tiles of both architectures' tables
        assert torch.equal(cuda_conv.launch_depthwise5x5_relu_pointwise(x, dw, dwb, pw, pwb, tile).permute(0, 3, 1, 2), got)
    tri = ops.depthwise5x5_relu_pointwise.get("triton")(x, dw, dwb, pw, pwb)
    assert got.shape == tri.shape and got.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(got, tri)
    ref = ops.depthwise5x5_relu_pointwise.get("reference")(x, dw, dwb, pw, pwb)
    assert (got != ref).float().mean() < 1e-3  # FP32 reduction order only, as the Triton op
    assert ((got.float() - ref.float()).abs() <= ref.float().abs() * 2**-7 + 1e-6).all()
    assert (ops.depthwise5x5_relu_pointwise.explain(x, dw, dwb, pw, pwb)["cuda"] == "selected") is _selected_on_measured_archs()


def test_depthwise_cuda_ops_compile_without_fallback():
    """The ``custom_op`` wrappers trace (fake kernels) and the compiled result matches eager."""
    x, w, b, s = _producer_inputs(1, 256, 24, 31)
    fn = torch.compile(lambda *a: ops.depthwise5x5_relu_int8.get("cuda")(*a), fullgraph=True)
    assert torch.equal(fn(x, w, b, s), ops.depthwise5x5_relu_int8.get("cuda")(x, w, b, s))
