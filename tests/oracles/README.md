# Vendored reference implementations

The parity tests (`tests/parity/*`) compare each FlashIM model with the authors' own code. That code is copied here,
restricted to the files the tests import, so the tests do not depend on checkouts elsewhere. Every `.py` file starts
with a header naming its origin, copyright and license; the upstream `LICENSE` sits in each directory. Files are
unmodified apart from that header, except `__init__.py` files marked "trimmed" (upstream's imports more than the tests
need). `pyproject.toml` excludes this directory from ruff. Model checkpoints are not here: they come from FlashIM's
weight lookup (`flashim/models/*/weights.py`).

| directory | upstream | license |
|---|---|---|
| `romav2/` | https://github.com/Parskatt/RoMaV2 @ 95c9968 (`vit/` and `dpt.py` carry Meta's DINOv3 License: `LICENSE_DINOv3`) | MIT |
| `dad/` | https://github.com/Parskatt/dad | MIT |
| `dedode/` | https://github.com/Parskatt/DeDoDe (`transformer/`: DINOv2, Meta) | MIT (DINOv2: Apache-2.0) |
| `aliked/` | https://github.com/Shiaoming/ALIKED | BSD-3-Clause |
| `alike/` | https://github.com/Shiaoming/ALIKE | BSD-3-Clause |
| `lightglue/` | https://github.com/cvg/LightGlue (`lightglue.py`, `utils.py`; not `superpoint.py`) | Apache-2.0 |
| `xfeat/` | https://github.com/verlab/accelerated_features | Apache-2.0 |
| `rdd/` | https://github.com/xtcpete/rdd @ 02ff372 (`convnext.py`: Meta, MIT: `LICENSE_ConvNeXt`; Deformable-DETR parts: SenseTime, Apache-2.0) | Apache-2.0 |
| `raco/` | https://github.com/fabio-sim/LightGlue-ONNX (from cvg/RaCo) | Apache-2.0 |

Not vendored: cvg/LightGlue's `lightglue/superpoint.py` (it carries a Magic Leap notice forbidding reproduction and
distribution) and RIPE. Both are downloaded on demand by their tests (`tests/parity/_fetch.py`). RIPE (https://github.com/fraunhoferhhi/RIPE) is not vendored: its license (Fraunhofer academic use, version 2.0) grants
internal non-commercial use only and no other rights, so its code cannot be shipped here. `tests/parity/ripe` downloads
the files it needs (pinned by commit and SHA-256) into `~/.cache/flashim/oracles` when you run it, or uses the checkout
in `$FLASHIM_RIPE_ROOT`; it skips without network access.
