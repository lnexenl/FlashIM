# Third-party notices

FlashIM's models are re-implementations of published methods (module and parameter names follow the upstream
checkpoints so they load with `strict=True`). A comparison of non-trivial lines against the upstream sources found
at most 5% verbatim matches in each model (10% for RaCo, 0% for SuperPoint and RIPE); the places below are where
upstream code or licenses nevertheless apply.

## Code in `flashim/`

- **RoMaV2**, https://github.com/Parskatt/RoMaV2, MIT, Copyright (c) 2025 Johan Edstedt (`licenses/RoMaV2-MIT.txt`).
  Triton kernels in `flashim/backends/triton/` (`attention.py`: RoPE; `conv.py`: refiner blocks; `quantize.py`:
  producer -> MXFP8 kernels; `gemm.py`: launch configuration) are ported from it, and `flashim/models/romav2/` follows its architecture.
- **DINOv3 / DPT code in RoMaV2**, Copyright (c) Meta Platforms, Inc. and affiliates, under the **DINOv3 License
  Agreement** (`licenses/DINOv3-License.md`): `flashim/models/romav2/dinov3.py` and `dpt.py` follow code that RoMaV2
  took from Meta. Redistributing them (or the DINOv3 weights inside the RoMaV2 checkpoint, which FlashIM downloads
  and does not ship) is subject to that agreement.
- **LightGlue**, https://github.com/cvg/LightGlue, Apache-2.0: `flashim/models/aliked/descriptor.py` ports its
  `get_patches` fallback; `flashim/models/lightglue/` follows it (and RDD's variant).

Other models (ALIKED and ALIKE: BSD-3-Clause; DaD, DeDoDe: MIT, DeDoDe's DINOv2 parts Apache-2.0; LoMa: MIT, its
LightGlue-derived matcher Apache-2.0, built on `flashim/models/lightglue/`; RDD, XFeat, LoFTR, DISK, RaCo: Apache-2.0)
are independent implementations of the published architectures; the license of each upstream is recorded in
`flashim/models/<model>/weights.py` (`LICENSES`) next to the checkpoint it applies to.

## Weights (downloaded, never shipped)

Checkpoints are fetched from their publishers (URLs and SHA-256 in `flashim/models/*/weights.py`) and keep their own
licenses. Two are not open-source licenses: **SuperPoint** (`superpoint_v1.pth`, Magic Leap, non-commercial research)
and **RIPE** (Fraunhofer academic-use license: internal non-commercial use only). The DINOv3 weights inside the RoMaV2
checkpoint are under the DINOv3 License.

## Test references (`tests/oracles/`)

Upstream reference code for the parity tests, copied with its license next to it and a header in each file; see
`tests/oracles/README.md`. RIPE and cvg/LightGlue's `superpoint.py` (Magic Leap notice) are not copied: the tests
download them on demand. `assets/` holds test images taken from RoMaV2 (MIT), MegaDepth and LightGlue-ONNX (Apache-2.0).

## Calibration tables

`flashim/models/*/calibration/*.safetensors` are activation scales measured by FlashIM on MegaDepth training pairs;
they contain no upstream weights.
