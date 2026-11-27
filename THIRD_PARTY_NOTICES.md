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
- **PyTorch / ATen**, https://github.com/pytorch/pytorch, BSD-3-Clause: `flashim/backends/triton/_layernorm_stats.py`
  re-expresses in Triton the arithmetic order of ATen's CUDA LayerNorm row statistics (`vectorized_layer_norm_kernel`:
  per-thread Welford update and combine, reduction tree) so fused kernels reproduce `nn.LayerNorm` bit for bit. No ATen
  source is copied.

Other models (ALIKED and ALIKE: BSD-3-Clause; DaD, DeDoDe, FlexGlue, RoMa, Tiny-RoMa: MIT, DINOv2 parts Apache-2.0, Tiny-RoMa's XFeat backbone Apache-2.0; LoMa: MIT, its
LightGlue-derived matcher Apache-2.0, built on `flashim/models/lightglue/`; OmniGlue and DINOv2 ViT-B/14: Apache-2.0,
with rpautrat/SuperPoint `sp_v6`: MIT; RDD, TraqPoint, XFeat, LoFTR, EfficientLoFTR @ `ffd4a46440`, XoFTR, DISK, RaCo, MINIMA: Apache-2.0;
UFM and UniCeption: BSD-3-Clause, CroCo DPT blocks CC BY-NC-SA 4.0, DINOv2 Apache-2.0;
MASt3R / DUSt3R / CroCo: CC BY-NC-SA 4.0, non-commercial use only; note that upstream `zju3dv/EfficientLoFTR` re-licensed to ZJU3DV PRL v1.0 in commit `07e9c1401e` on 2026-09-15)
are independent implementations of the published architectures; the license of each upstream is recorded in
`flashim/models/<model>/weights.py` (`LICENSES`) next to the checkpoint it applies to.

## Weights (downloaded, never shipped)

Checkpoints are fetched from their publishers (URLs and SHA-256 in `flashim/models/*/weights.py`) and keep their own
licenses. Seven families are not open-source licenses: **SuperPoint** (`superpoint_v1.pth`, Magic Leap, non-commercial research),
**SuperGlue** (`superglue_outdoor.pth` / `superglue_indoor.pth`, Magic Leap, academic or non-profit non-commercial
research use only), **RIPE** (`ripe_weights.pth`, Fraunhofer academic-use license v2.0: internal non-commercial use only),
**RIPE++** (`ripe++.ckpt`, `ripe++_tokyo_megadepth.ckpt`, `ripe++_scared.ckpt`, and `ripe++lightglue.tar`, Fraunhofer
academic-use license v2.0: internal non-commercial use only; upstream repo partly based on DALF Apache-2.0, DeDoDe MIT,
DISK Apache-2.0, and matcher code in `JohannesK14/glue-factory` Apache-2.0), **MASt3R**
(`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`, Naver Corporation, CC BY-NC-SA 4.0, non-commercial use only),
**DUSt3R** (`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`, Naver Corporation, CC BY-NC-SA 4.0, non-commercial use only),
and **UFM** (`ufm_base.safetensors`, CC BY-NC-SA 4.0, non-commercial use only).
The DINOv3 weights inside the RoMaV2 checkpoint, the DINOv3 ConvNeXt-B backbone inside FlexGlue's `flexglue.pth`
(`extractor.backbone.*`), and the fine-tuned DINOv3 ConvNeXt-B backbone inside TraqPoint's `traqpoint_best.pth`
(`descriptor.backbone.0.dino.*`) are under the DINOv3 License (`licenses/DINOv3-License.md`).

## Test references (`tests/oracles/`)

Upstream reference code for the parity tests, copied with its license next to it and a header in each file; see
`tests/oracles/README.md`. RIPE, RIPE++, cvg/LightGlue's `superpoint.py` and Magic Leap's `superglue.py` (Magic Leap notice)
are not copied: the tests download them on demand (or import an installed package / local checkout). `assets/` holds test images taken from RoMaV2 (MIT), MegaDepth and LightGlue-ONNX (Apache-2.0).

## Calibration tables

`flashim/models/*/calibration/*.safetensors` are activation scales measured by FlashIM on MegaDepth training pairs;
they contain no upstream weights.
