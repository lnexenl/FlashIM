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
| `roma/` | https://github.com/Parskatt/RoMa (`romatch/models/transformer/`: DINOv2, Meta: `LICENSE_DINOv2`) | MIT (DINOv2: Apache-2.0) |
| `tiny_roma/` | https://github.com/Parskatt/RoMa (`romatch/models/tiny.py`, `romatch/utils/kde.py`) | MIT |
| `dad/` | https://github.com/Parskatt/dad | MIT |
| `dedode/` | https://github.com/Parskatt/DeDoDe (`transformer/`: DINOv2, Meta) | MIT (DINOv2: Apache-2.0) |
| `aliked/` | https://github.com/Shiaoming/ALIKED | BSD-3-Clause |
| `alike/` | https://github.com/Shiaoming/ALIKE | BSD-3-Clause |
| `lightglue/` | https://github.com/cvg/LightGlue (`lightglue.py`, `utils.py`; not `superpoint.py`) | Apache-2.0 |
| `xfeat/` | https://github.com/verlab/accelerated_features | Apache-2.0 |
| `rdd/` | https://github.com/xtcpete/rdd @ 02ff372 (`convnext.py`: Meta, MIT: `LICENSE_ConvNeXt`; Deformable-DETR parts: SenseTime, Apache-2.0) | Apache-2.0 |
| `raco/` | https://github.com/fabio-sim/LightGlue-ONNX (from cvg/RaCo) | Apache-2.0 |
| `flexglue/` | https://github.com/cvg/FlexGlue @ c6cbc7a (`flexglue/`, not `utils/viz2d.py`; `refiner.py`: from RoMaV2, MIT; `LICENSE_DINOv3.md`: the DINOv3 License of the checkpoint's backbone weights, which are not here) | MIT |
| `loma/` | https://github.com/davnords/LoMa @ 415caed (`src/loma/`; `descriptor/transformer/`: DINOv2, Meta; offline loader `loma_oracle.py`, see `loma/README.md`) | MIT (matcher in `loma.py`: Apache-2.0, from LightGlue; DINOv2: Apache-2.0) |
| `eloftr/` | https://github.com/zju3dv/EfficientLoFTR @ ffd4a46440 (`src/loftr/`, `src/utils/misc.py`; last Apache-2.0 commit before upstream re-licensed to ZJU3DV PRL v1.0 in `07e9c1401e`) | Apache-2.0 (`repvgg.py`: RepVGG MIT) |
| `xoftr/` | https://github.com/OnderT/XoFTR (`src/xoftr/` + `src/config/default.py`, trimmed `xoftr/__init__.py`) | Apache-2.0 |
| `minima/` | https://github.com/LSXI7/MINIMA (`LoFTR_minima` submodule `src/loftr/`; `minima-roma` parity reuses `roma/`) | Apache-2.0 |
| `mast3r/` | https://github.com/naver/mast3r + submodules https://github.com/naver/dust3r and https://github.com/naver/croco | CC BY-NC-SA 4.0 (non-commercial use only) |
| `dust3r/` | https://github.com/naver/dust3r + submodule https://github.com/naver/croco | CC BY-NC-SA 4.0 (non-commercial use only) |
| `ufm/` | https://github.com/UniFlowMatch/UFM + https://github.com/castacks/UniCeption @ ee7fa0b (`uniflowmatch/`, `uniception/`; offline loader `ufm_oracle.py`, see `ufm/README.md`) | BSD-3-Clause (`croco/` DPT blocks: CC BY-NC-SA 4.0) |
| `omniglue/` | https://github.com/google-research/omniglue (`src/omniglue/`, `third_party/dinov2/`; TF-CPU subprocess runner `tf_oracle.py`, set `$FLASHIM_OMNIGLUE_TF_PYTHON`, see `omniglue/README.md`) | Apache-2.0 |

Not vendored: cvg/LightGlue's `lightglue/superpoint.py`, Magic Leap's `SuperGluePretrainedNetwork` (`models/superglue.py`,
both carry a Magic Leap notice forbidding reproduction and distribution), RIPE, and RIPE++. All four are downloaded on
demand by their tests (`tests/parity/_fetch.py`). RIPE (https://github.com/fraunhoferhhi/RIPE) and RIPE++
(https://github.com/fraunhoferhhi/RIPEpp) are not vendored: their license (Fraunhofer academic use, version 2.0) grants
internal non-commercial use only and no other rights, so their code cannot be shipped here. `tests/parity/ripe` and
`tests/parity/ripepp` download the files they need (pinned by commit and SHA-256) into `~/.cache/flashim/oracles` when
you run them, or use an existing checkout in `$FLASHIM_RIPE_ROOT` / `$FLASHIM_RIPEPP_ROOT` (or an installed `ripepp`
package); they skip without network access.
