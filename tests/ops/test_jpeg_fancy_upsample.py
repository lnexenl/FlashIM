"""``ops.jpeg_fancy_upsample``: the reference against a literal transcription of libjpeg-turbo's C code (jdsample.c
``h2v1_fancy_upsample`` / ``h2v2_fancy_upsample``, jdcolor.c ``ycc_rgb_convert``) on synthetic YCbCr planes; the
Triton impl against the reference; the GPU decoder against PIL on the asset JPEGs."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flashim import ops

ASSETS = Path(__file__).resolve().parents[2] / "assets"


def _fix(x):
    return int(x * 65536 + 0.5)


def _ycc_to_rgb(y, cb, cr):
    """jdcolor.c: ``build_ycc_rgb_table`` + ``ycc_rgb_convert`` (with ``range_limit``)."""
    x = np.arange(256) - 128
    cr_r = (_fix(1.402) * x + 32768) >> 16
    cb_b = (_fix(1.772) * x + 32768) >> 16
    cr_g, cb_g = -_fix(0.71414) * x, -_fix(0.34414) * x + 32768
    rgb = np.stack([y + cr_r[cr], y + ((cb_g[cb] + cr_g[cr]) >> 16), y + cb_b[cb]])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _h2v1(c):
    out = np.zeros((c.shape[0], 2 * c.shape[1]), np.int64)
    for i, row in enumerate(c.astype(np.int64)):
        n, o = len(row), out[i]
        o[0], o[1] = row[0], (row[0] * 3 + row[1] + 2) >> 2
        for k in range(1, n - 1):
            o[2 * k], o[2 * k + 1] = (row[k] * 3 + row[k - 1] + 1) >> 2, (row[k] * 3 + row[k + 1] + 2) >> 2
        o[2 * n - 2], o[2 * n - 1] = (row[n - 1] * 3 + row[n - 2] + 1) >> 2, row[n - 1]
    return out


def _h2v2(c):
    c = c.astype(np.int64)
    m, n = c.shape
    ctx = np.concatenate([c[:1], c, c[-1:]])  # jdmainct.c: context rows replicate the first / last row
    out = np.zeros((2 * m, 2 * n), np.int64)
    for i in range(m):
        for v in range(2):
            col = ctx[i + 1] * 3 + (ctx[i] if v == 0 else ctx[i + 2])
            o = out[2 * i + v]
            o[0], o[1] = (col[0] * 4 + 8) >> 4, (col[0] * 3 + col[1] + 7) >> 4
            for k in range(1, n - 1):
                o[2 * k], o[2 * k + 1] = (col[k] * 3 + col[k - 1] + 8) >> 4, (col[k] * 3 + col[k + 1] + 7) >> 4
            o[2 * n - 2], o[2 * n - 1] = (col[n - 1] * 3 + col[n - 2] + 8) >> 4, (col[n - 1] * 4 + 7) >> 4
    return out


def _synthetic(h, w, v_factor, seed, spread):
    """(nvJPEG-like decode with replicated chroma, libjpeg's fancy decode) of random YCbCr planes."""
    rng = np.random.default_rng(seed)
    hb, wb = -(-h // v_factor), -(-w // 2)
    y = rng.integers(128 - spread, 128 + spread, (h, w))
    cb, cr = (rng.integers(128 - spread // 2, 128 + spread // 2, (hb, wb)) for _ in range(2))
    rep = lambda c: c.repeat(v_factor, 0).repeat(2, 1)[:h, :w]
    up = _h2v2 if v_factor == 2 else _h2v1
    replicated = _ycc_to_rgb(y, rep(cb), rep(cr))
    fancy = _ycc_to_rgb(y, up(cb)[:h, :w], up(cr)[:h, :w])
    return torch.from_numpy(replicated), torch.from_numpy(fancy)


SHAPES = [(8, 8), (9, 7), (16, 33), (31, 18), (3, 3)]


@pytest.mark.parametrize("v_factor", [1, 2])
@pytest.mark.parametrize("shape", SHAPES)
def test_reference_matches_libjpeg(shape, v_factor):
    """No clipping (Y within 68..187, chroma within 128 +- 30): exact."""
    replicated, fancy = _synthetic(*shape, v_factor, seed=0, spread=60)
    assert torch.equal(ops.jpeg_fancy_upsample.reference(replicated, v_factor), fancy)


@pytest.mark.parametrize("v_factor", [1, 2])
def test_reference_with_clipping(v_factor):
    """Full-range planes clip often; still far closer to libjpeg than the replicated decode."""
    replicated, fancy = _synthetic(64, 64, v_factor, seed=1, spread=128)
    out = ops.jpeg_fancy_upsample.reference(replicated, v_factor)
    err = (out.int() - fancy.int()).abs().float().mean()
    assert err < 0.25 * (replicated.int() - fancy.int()).abs().float().mean()


def _triton_available() -> bool:
    return any(i.name == "triton" for i in ops.jpeg_fancy_upsample.impls)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("v_factor", [1, 2])
@pytest.mark.parametrize("shape", [*SHAPES, (608, 821), (1199, 1600)])
@pytest.mark.parametrize("spread", [60, 128])
def test_triton_matches_reference(shape, v_factor, spread):
    assert _triton_available()
    replicated, _ = _synthetic(*shape, v_factor, seed=2, spread=spread)
    replicated = replicated.cuda()
    assert ops.jpeg_fancy_upsample.explain(replicated, v_factor)["triton"] == "selected"
    fast = ops.jpeg_fancy_upsample.get("triton")(replicated, v_factor)
    assert torch.equal(fast, ops.jpeg_fancy_upsample.reference(replicated, v_factor))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", ["sacre_coeur1.jpg", "toronto_A.jpg", "megadepth_0015/299056923_ecd980298c_o.jpg"])
def test_gpu_decoder_close_to_pil(name):
    from flashim.models.romav2.model import load_image

    path, device = ASSETS / name, torch.device("cuda")
    ref = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).to(device).int()
    raw = (load_image(path, device, "nvjpeg-raw")[0] * 255).round().int()
    out = (load_image(path, device, "nvjpeg")[0] * 255).round().int()
    raw_err, err = (raw - ref).abs().float().mean(), (out - ref).abs().float().mean()
    assert raw_err > 0.1  # subsampled chroma: nvJPEG's replication is visible
    # residual: chroma blocks where every pixel clips (toronto_A: 8% clipped pixels, 0.57 -> 0.03 levels on average)
    assert err < 0.05 and err < raw_err / 10


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_decoder_444_and_gray_identical(tmp_path):
    from flashim.models.romav2.model import load_image

    device = torch.device("cuda")
    image = Image.open(ASSETS / "sacre_coeur1.jpg").convert("RGB")
    image.save(tmp_path / "s444.jpg", subsampling=0)
    image.convert("L").save(tmp_path / "gray.jpg")
    image.convert("CMYK").save(tmp_path / "cmyk.jpg")  # nvJPEG cannot decode it: PIL's path
    for name in ("s444.jpg", "gray.jpg", "cmyk.jpg"):
        assert torch.equal(load_image(tmp_path / name, device, "nvjpeg"), load_image(tmp_path / name, device, "pil"))
