"""The image-path decoder switch of ``load_image`` (CPU): CPU loads are PIL's whatever the decoder, unknown decoders
are rejected, ``IMAGE_DECODER`` is the default, and ``chroma_layout`` routes each JPEG kind."""

import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from flashim.image_io import chroma_layout
from flashim.models.romav2 import model as romav2_model
from flashim.models.romav2.model import IMAGE_DECODERS, load_image

ASSET = Path(__file__).resolve().parents[2] / "assets" / "sacre_coeur1.jpg"
CPU = torch.device("cpu")


@pytest.mark.parametrize("decoder", IMAGE_DECODERS)
def test_cpu_loads_are_pil(decoder):
    assert torch.equal(load_image(ASSET, CPU, decoder), load_image(ASSET, CPU, "pil"))
    assert load_image(ASSET, CPU, decoder).shape == (1, 3, 731, 1024)


def test_unknown_decoder():
    with pytest.raises(ValueError, match="Unknown image decoder"):
        load_image(ASSET, CPU, "libjpeg")


def test_default_is_nvjpeg():
    assert romav2_model.IMAGE_DECODER == os.environ.get("FLASHIM_IMAGE_DECODER", "nvjpeg")


def test_default_decoder(monkeypatch):
    monkeypatch.setattr(romav2_model, "IMAGE_DECODER", "nope")
    with pytest.raises(ValueError, match="nope"):
        load_image(ASSET, CPU)
    monkeypatch.setattr(romav2_model, "IMAGE_DECODER", "nvjpeg")
    assert torch.equal(load_image(ASSET, CPU), load_image(ASSET, CPU, "pil"))


def test_chroma_layout(tmp_path):
    image = Image.open(ASSET).convert("RGB")
    cases = {"s420.jpg": ({"subsampling": 2}, 2), "s422.jpg": ({"subsampling": 1}, 1),
             "s444.jpg": ({"subsampling": 0}, 0), "prog.jpg": ({"subsampling": 2, "progressive": True}, 2)}
    for name, (kwargs, layout) in cases.items():
        image.save(tmp_path / name, **kwargs)
        assert chroma_layout((tmp_path / name).read_bytes()) == layout, name
    image.convert("L").save(tmp_path / "gray.jpg")
    image.convert("CMYK").save(tmp_path / "cmyk.jpg")
    image.save(tmp_path / "png.jpg", format="PNG")  # misnamed: not a JPEG
    assert chroma_layout((tmp_path / "gray.jpg").read_bytes()) == 0
    assert chroma_layout((tmp_path / "cmyk.jpg").read_bytes()) is None
    assert chroma_layout((tmp_path / "png.jpg").read_bytes()) is None
    assert chroma_layout(b"not an image") is None
