"""GPU decoding of JPEG files for ``load_image`` (``flashim.models.romav2.model``), as close to PIL's as it gets.

nvJPEG (``torchvision.io.decode_jpeg(..., device="cuda")``) matches libjpeg-turbo (PIL) bit for bit on 4:4:4 and
grayscale JPEGs (all 144 such MegaDepth-1500 images), but upsamples subsampled chroma (4:2:2, 4:2:0) by pixel
replication where libjpeg's default "fancy" upsampling is a triangle filter: over the 806 MegaDepth-1500 images 0.18
levels apart on average, up to 43. ``ops.jpeg_fancy_upsample`` redoes the upsampling libjpeg's way on the GPU: 0.0008 on
average, at most 6, 0.07% of the pixels differ; what is left is in chroma blocks where every pixel clips a channel
(saturated sky). Measured on L20 and H20 (identical) with ``benchmarks/speed/image_decode.py``.
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image, JpegImagePlugin
from torch import Tensor

from flashim import ops
from flashim.log import get_logger, warn_once

logger = get_logger(__name__)

JPEG_SUFFIXES = (".jpg", ".jpeg")


def chroma_layout(data: bytes) -> int | None:
    """How nvJPEG output of this JPEG relates to PIL's: 0 = identical layout (grayscale or 4:4:4 YCbCr), 1 / 2 =
    4:2:2 / 4:2:0 YCbCr (needs ``ops.jpeg_fancy_upsample`` with that ``v_factor``), ``None`` = decode it with PIL
    (CMYK, RGB-coded, other subsamplings, non-JPEG)."""
    try:
        image = Image.open(io.BytesIO(data))
    except OSError:
        return None
    if image.format != "JPEG":
        return None
    if image.mode == "L":
        return 0
    if image.mode != "RGB" or image.info.get("adobe_transform") == 0 or [c[0] for c in image.layer] != [1, 2, 3]:
        return None
    return {0: 0, 1: 1, 2: 2}.get(JpegImagePlugin.get_sampling(image))


def decode_jpeg_cuda(path: str | Path, device: torch.device, fancy: bool = True) -> Tensor | None:
    """``[3, H, W]`` uint8 on ``device`` (nvJPEG; ``fancy``: libjpeg's chroma upsampling), or ``None`` when the file
    should go to PIL instead (see :func:`chroma_layout`)."""
    from torchvision.io import ImageReadMode, decode_jpeg

    data = Path(path).read_bytes()
    layout = chroma_layout(data)
    if layout is None:
        return None
    try:
        rgb = decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB, device=device)
    except (RuntimeError, NotImplementedError) as exc:  # e.g. arithmetic coding, 12-bit, torchvision without nvJPEG
        warn_once(logger, "nvjpeg-fallback", "nvJPEG cannot decode %s (%s); using PIL", path, str(exc).splitlines()[0])
        return None
    return ops.jpeg_fancy_upsample(rgb, layout) if fancy and layout else rgb
