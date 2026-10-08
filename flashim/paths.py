"""Where the tests, benchmarks and scripts find their inputs.

``ASSETS``: the small images checked in under ``assets/``. ``DATA``: the (untracked) datasets, ``data/`` next to the
repository or ``$FLASHIM_DATA``: ``megadepth/``, ``scannet/``, ``quantization/`` (calibration manifests and the
RoMaV2 oracle's ``romav2_full_fp8_mega1500.pt``), ``hardmatch/``.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"
DATA = Path(os.environ.get("FLASHIM_DATA", REPO / "data")).expanduser()
MEGADEPTH_IMAGES = ASSETS / "megadepth_0015"
IMAGE_A = MEGADEPTH_IMAGES / "3383162411_67cc9af85e_o.jpg"
IMAGE_B = MEGADEPTH_IMAGES / "299056923_ecd980298c_o.jpg"
ROMAV2_CALIBRATION = DATA / "quantization" / "romav2_full_fp8_mega1500.pt"
