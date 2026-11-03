"""``ops.pil_bicubic_resize`` (Triton) against its torch reference, bit for bit."""

import pytest
import torch

from flashim import ops

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
from flashim.backends.triton import resample as triton_resample

pytestmark = pytest.mark.cuda
DEV = "cuda"


def _both(op, impl, *args, **kwargs):
    return op.get(impl)(*args, **kwargs), op.get("reference")(*args, **kwargs)


_PIL_RESIZE_CASES = [
    ((1200, 1600), [(768, 1024), (784, 784)]),  # LoMa's two sizes of a MegaDepth landscape image
    ((1600, 1063), [(1024, 680), (784, 784)]),  # portrait: 2.04x downscale in height (11 taps)
    ((37, 45), [(16, 24), (100, 13)]),  # odd sizes, down in both / up in height
    ((300, 400), [(384, 520)]),  # up in both, one size
    ((768, 1000), [(768, 784), (500, 1000)]),  # one side kept: that pass is skipped (no rounding)
    ((5, 7), [(3, 2), (11, 13), (5, 7)]),  # tiny, three sizes (two launches), identity
    ((2000, 3000), [(680, 1024), (784, 784)]),
]


@pytest.mark.parametrize("src,sizes", _PIL_RESIZE_CASES)
@pytest.mark.parametrize("kind", ["uint8", "uint8_hwc", "float"])
def test_pil_bicubic_resize(src, sizes, kind):
    """Bit-identical to the reference's torch passes: the same span / weight / FMA arithmetic and 8-bit roundings."""
    torch.manual_seed(0)
    if kind == "float":  # off the 8-bit grid, batch 2: the roundings and skipped passes matter
        image = torch.rand(2, 3, *src, device=DEV)
    else:
        image = torch.randint(0, 256, (1, 3, *src), dtype=torch.uint8, device=DEV)
        if kind == "uint8_hwc":  # load_image_raw's layout of a decoded image (a permuted HWC array)
            image = image.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    fast, ref = _both(ops.pil_bicubic_resize, "triton", image, sizes)
    assert len(fast) == len(ref) == len(sizes)
    for f, r, (h, w) in zip(fast, ref, sizes):
        assert f.shape == r.shape == (image.shape[0], 3, h, w) and f.dtype == torch.float32
        assert torch.equal(f.view(torch.int32), r.view(torch.int32))


def test_pil_bicubic_resize_real_image():
    """A MegaDepth image at LoMa's detector (long side 1024) and descriptor (784 x 784) sizes, as extract calls it."""
    from flashim.models.loma.extractor import detector_size
    from flashim.models.romav2.model import load_image_raw
    from flashim.paths import IMAGE_A

    image = load_image_raw(IMAGE_A, torch.device(DEV))
    sizes = [detector_size(*image.shape[-2:], 1024), (784, 784)]
    assert ops.pil_bicubic_resize.explain(image, sizes)["triton"] == "selected"
    fast, ref = _both(ops.pil_bicubic_resize, "triton", image, sizes)
    for f, r in zip(fast, ref):
        assert torch.equal(f.view(torch.int32), r.view(torch.int32))


@triton.jit
def _division_kernel(a_ptr, b_ptr, fast_ptr, exact_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a, b = tl.load(a_ptr + i), tl.load(b_ptr + i)
    rcp = tl.math.div_rn(tl.full([BLOCK], 1.0, tl.float32), b)
    tl.store(fast_ptr + i, triton_resample._div_rn(a, b, rcp))
    tl.store(exact_ptr + i, tl.math.div_rn(a, b))


def test_pil_resize_division_is_correctly_rounded():
    """The height weights' reciprocal + two-FMA-correction division equals ``div_rn`` bit for bit over (wider than)
    its operand range: bicubic values (incl. tiny ones near the filter's ends) over weight sums."""
    torch.manual_seed(0)
    n = 1 << 24
    a = (torch.rand(n, device=DEV) * 2.2 - 1.1) * torch.pow(10.0, -torch.randint(0, 13, (n,), device=DEV).float())
    b = torch.pow(2.0, torch.rand(n, device=DEV) * 6 - 2)  # [0.25, 16)
    fast, exact = torch.empty_like(a), torch.empty_like(a)
    _division_kernel[(n // 1024,)](a, b, fast, exact, BLOCK=1024)
    assert torch.equal(fast.view(torch.int32), exact.view(torch.int32))
