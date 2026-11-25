<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lnexenl/FlashIM/main/assets/brand/lockup-dark@8x.png">
    <img alt="FlashIM" src="https://raw.githubusercontent.com/lnexenl/FlashIM/main/assets/brand/lockup-light@8x.png" width="560">
  </picture>
</p>

<p align="center">
  <b>Fast, quantized, multi-platform inference for image matching.</b><br>
  Dense and sparse matchers as native PyTorch ports, with FP8 / INT8 quantization and custom kernels.
</p>

<p align="center">
  <a href="https://github.com/lnexenl/FlashIM/blob/main/LICENSE">Apache-2.0</a> ·
  <a href="https://github.com/lnexenl/FlashIM/blob/main/docs/models.md">Models and results</a> ·
  <a href="https://github.com/lnexenl/FlashIM/blob/main/docs/benchmarks.md">Inference benchmark</a> ·
  <a href="https://github.com/lnexenl/FlashIM/blob/main/docs/configuration.md">Configuration</a> ·
  <a href="https://github.com/lnexenl/FlashIM/blob/main/docs/platforms.md">Platforms</a> ·
  <a href="https://github.com/lnexenl/FlashIM/blob/main/docs/architecture.md">Architecture</a>
</p>

FlashIM runs image-matching models (dense RoMa, RoMa v2, Tiny-RoMa, MASt3R, DUSt3R, UFM; detector-free LoFTR, EfficientLoFTR and XoFTR; sparse RDD, ALIKED, RaCo, SuperPoint, DISK, DeDoDe and others; LightGlue, SuperGlue, LoMa and OmniGlue matching) faster than their reference
implementations, without giving up measured accuracy. Every model is a native PyTorch port checked against the upstream code, and the
speedups come from quantization (MXFP8, INT8), fused kernels (Triton, TileLang, CUDA C++) and `torch.compile`.

- **Custom kernels are first-class.** Every op has a torch reference (the correctness oracle and universal fallback) and optional
  optimized implementations that are picked by platform and shape.
- **Quantization with evidence.** FP8 / INT8 presets are accepted by task-level pose AUC (MegaDepth-1500, ScanNet-1500, HardMatch), not by
  tensor `allclose`.
- **Multi-platform by design.** Device detection separates vendor, dispatch key and ISA features. NVIDIA from Ampere to Blackwell today; ROCm, NPU and XPU are planned.
- **One interface.** Every model loads with `flashim.load(name, preset)`, where the preset is `fp32`, `bf16`, `int8` or `fp8`, and matches with
  `model.match(a, b)`, which returns the same `Matches` for dense, detector-free and sparse models.
- **One builder, declarative config.** A `RunConfig` (arch, precision, optimization passes, engine) is layered, hashable and reproducible.

## Quick start

FlashIM is not on PyPI yet; install from source (Linux, Python 3.10 or newer, an NVIDIA GPU is recommended):

```bash
pip install "flashim[cv] @ git+https://github.com/lnexenl/FlashIM"   # or, in a checkout: uv sync --extra eval
```

The `cv` extra adds OpenCV (SIFT, ORB), `eval` the accuracy benchmarks' dependencies. Jetson (JetPack 6) installs from
[`requirements-jetson.txt`](https://github.com/lnexenl/FlashIM/blob/main/requirements-jetson.txt) (NVIDIA's torch build).

```python
import flashim

model = flashim.load("romav2", "fp8", engine="compile-full", overrides={"arch.setting": "mega1500"})
m = model.match("a.jpg", "b.jpg", max_matches=5000)          # Matches: 5000 balanced samples of the dense warp
m.keypoints_A, m.keypoints_B, m.confidence                   # [N, 2] pixels of each original image, [N]
m.dense.warp_AB                                              # the full dense prediction (DenseMatchResult)

pipeline = flashim.load("raco-aliked-lightglue", "bf16", engine="compile")
m = pipeline.match("a.jpg", "b.jpg", resize=1024, max_matches=1024)   # Matches: 1024 keypoints per image
m.indices_A, m.features_A                                    # match indices into each image's keypoints

loma = flashim.load("loma-g")                                # LoMa: DaD + LoMa descriptors + matcher (b128, b, l, g, r)
m = loma.match("a.jpg", "b.jpg", max_matches=4096)           # 4096 keypoints per image (default 2048)

roma = flashim.load("roma")                                  # RoMa (v1) dense matcher (outdoor; "roma-indoor", "minima-roma")
m = roma.match("a.jpg", "b.jpg", max_matches=5000)           # 5000 balanced samples of the dense warp (560 -> 864)

tiny = flashim.load("tiny-roma", "int8")                     # Tiny-RoMa dense matcher (fp32, bf16, int8, fp8)
m = tiny.match("a.jpg", "b.jpg", max_matches=5000)           # 5000 balanced samples of the dense warp

eloftr = flashim.load("eloftr")                              # EfficientLoFTR (bf16, reparameterized; variant full | opt)
m = eloftr.match("a.jpg", "b.jpg", resize=832)               # semi-dense matches (sides floored to multiples of 32)

xoftr = flashim.load("xoftr")                                # XoFTR: cross-modal visible-thermal matcher (default 640 px)
m = xoftr.match("a.jpg", "b.jpg")                            # Matches at original image resolution

minima = flashim.load("minima-loftr")                        # MINIMA modality-invariant LoFTR ("minima-roma" for RoMa)
m = minima.match("a.jpg", "b.jpg", resize=840)               # cross-modal & RGB semi-dense matches

sg = flashim.load("superpoint-superglue")                    # SuperPoint + SuperGlue (outdoor default; variant indoor)
m = sg.match("a.jpg", "b.jpg", resize=1024)                  # 1024 keypoints per image, match_threshold=0.2

mast3r = flashim.load("mast3r")                              # MASt3R: 3D-grounded dense matcher (512 px long side)
m = mast3r.match("a.jpg", "b.jpg")                           # Matches from fast reciprocal NNs on 24-D descriptors

dust3r = flashim.load("dust3r")                              # DUSt3R: stereo 3D pointmap matcher (512 px long side)
m = dust3r.match("a.jpg", "b.jpg")                           # Matches from exact reciprocal NNs in 3D pointmaps

ufm = flashim.load("ufm", "int8")                            # UFM: DINOv2 ViT-L + global ViT + dual DPT dense flow
m = ufm.match("a.jpg", "b.jpg", max_matches=5000)            # 5000 balanced samples of the dense flow + covisibility

og = flashim.load("omniglue")                                # OmniGlue: SuperPoint (sp_v6) + DINOv2 ViT-B/14 + matcher
m = og.match("a.jpg", "b.jpg", threshold=0.02)               # 1024 keypoints per image (default)
```

`match(a, b, *, resize, max_matches, threshold)` is the same for every model: `resize` is the long side of the working
resolution, `max_matches` bounds the matches the model's own way (keypoints per image, the most confident matches, or
samples of a dense warp) and `threshold` is the model's match threshold. A model without the preset asked for builds the
nearest one it has (`fp8` -> `int8` -> `bf16` -> `fp32`), and a pipeline's preset applies to each of its components.
Image paths given to a model on CUDA are decoded on the GPU when they are JPEGs (nvJPEG, with libjpeg's chroma
upsampling: 0.0008 levels from PIL on average, ~10 ms less per MegaDepth image); `FLASHIM_IMAGE_DECODER=pil` decodes with PIL.

Weights download on first use and are checked against pinned SHA-256 hashes ([Checkpoints](https://github.com/lnexenl/FlashIM#checkpoints)).

## Supported models

Default preset in **bold**; full registry and notes in [docs/models.md](https://github.com/lnexenl/FlashIM/blob/main/docs/models.md). Presets are always named `fp32` / `bf16` / `int8` / `fp8`; `mast3r` and `dust3r` run **FP16** compute under `bf16` / `int8` / `fp8` (CroCo residual activations reach ~344–649 where BF16 ULP degrades 3D precision).

| category | models | presets | upstream / weight license |
|---|---|---|---|
| Dense | `romav2` | `bf16`, `int8`, **`fp8`** | MIT + **DINOv3 License** (weights) |
| Dense | `roma`, `roma-indoor`, `minima-roma` | `fp32`, **`bf16`**, `int8`, `fp8` | MIT / Apache-2.0 (`minima-roma`); DINOv2 Apache-2.0 |
| Dense | `tiny-roma` | `fp32`, **`bf16`**, `int8`, `fp8` | MIT (XFeat backbone Apache-2.0) |
| Dense | `ufm` | `fp32`, **`bf16`**, `int8`, `fp8` | BSD-3-Clause code; **CC BY-NC-SA 4.0 weights (non-commercial)** |
| Dense 3D | `mast3r`, `dust3r` *(FP16 compute)* | `fp32`, **`bf16`**, `int8`, `fp8` | **CC BY-NC-SA 4.0 (non-commercial)** |
| Detector-free | `loftr`, `minima-loftr` | `fp32`, **`bf16`**, `int8`, `fp8` | Apache-2.0 |
| Detector-free | `eloftr` | `fp32`, **`bf16`**, `int8`, `fp8` | Apache-2.0 (`vismatch` weights & `ffd4a46440` oracle; upstream re-licensed to ZJU3DV PRL v1.0 on 2026-09-15) |
| Detector-free | `xoftr` | `fp32`, **`bf16`**, `int8`, `fp8` | Apache-2.0 |
| Sparse extractors | `rdd` | `fp32`, `bf16`, **`fp8`** | Apache-2.0 |
| Sparse extractors | `aliked`, `raco`, `raco-aliked` | **`fp32`**, `bf16` | BSD-3-Clause (`aliked`) / Apache-2.0 (`raco`) |
| Sparse extractors | `alike`, `disk`, `xfeat`, `dedode`, `dedode-g`, `dad` | `fp32`, **`bf16`**, `int8`, `fp8` *(`dedode` degrades `fp8`->`int8`)* | BSD-3-Clause / Apache-2.0 / MIT |
| Sparse extractors | `superpoint`, `ripe` | `fp32`, **`bf16`**, `int8`, `fp8` | **Non-commercial** (`superpoint_v1.pth` Magic Leap; `ripe` Fraunhofer; SuperPoint `sp_v6` is MIT) |
| Sparse extractors | `sift`, `orb` | **`fp32`** | Apache-2.0 (OpenCV) |
| Sparse matchers & pipelines | `lightglue`, `{rdd,aliked,raco-aliked,superpoint,disk,xfeat}-lightglue` | `fp32`, `bf16`, `int8`, **`fp8`** *(`aliked`/`superpoint`/`disk-lightglue` degrade `fp8`->`int8`; `xfeat-lightglue` degrades `int8`/`fp8`->`bf16`)* | Apache-2.0 (inherits extractor weight license) |
| Sparse matchers & pipelines | `loma`, `loma-{b128,b,l,g,r}`, `loma-{extractor,matcher}` | `fp32`, **`bf16`**, `int8`, `fp8` | MIT / Apache-2.0 |
| Sparse matchers & pipelines | `superglue`, `superpoint-superglue` | `fp32`, **`bf16`**, `int8` *(pipeline degrades `fp8`->`int8`)* | **Magic Leap (academic/non-profit non-commercial)** |
| Sparse matchers & pipelines | `omniglue`, `omniglue-{extractor,matcher}` | `fp32`, **`bf16`**, `int8` *(extractor adds `fp8`; pipeline & matcher degrade `fp8`->`int8`)* | Apache-2.0 (`sp_v6` MIT) |

## Results

All numbers below are from an **NVIDIA GeForce RTX 5070 Ti (`sm_120`)**. Full latency tables (including the 2026-11
`sm_120` eager speedup sweep) and MegaDepth-1500 / ScanNet-1500 accuracy gate results (`0 FAIL` across all models) are
in the [inference benchmark](https://github.com/lnexenl/FlashIM/blob/main/docs/benchmarks.md); models, presets and
per-architecture verification status are in [docs/models.md](https://github.com/lnexenl/FlashIM/blob/main/docs/models.md),
[docs/platforms.md](https://github.com/lnexenl/FlashIM/blob/main/docs/platforms.md) and each model's page.

**RoMaV2** (dense; `mega1500` 800² + 1024², both directions, forward only, `engine="compile-full"`, RTX 5070 Ti `sm_120`):

| preset | forward | vs `bf16` |
|---|---|---|
| `bf16` | 195 ms | 1.00x |
| `fp8` | 110 ms | 1.77x |
| `int8` | 104 ms | 1.87x |

In eager mode on RTX 5070 Ti (`sm_120`, `mega1500`, both directions), `fp8` runs in 113.9 ms (1.62x vs pre-sweep `fp8`)
and `int8` in 110.8 ms (1.26x). Pose AUC@5 vs `bf16` (3 seeds, MegaDepth-1500 / ScanNet-1500, `sm_120`, `compile-full`):
`fp8` -0.0023 / +0.0007, `int8` -0.0010 / +0.0004, inside the RANSAC seed noise of about 0.003 (and 0 FAIL under the
2026-11 `compile` paired-bootstrap gate). On the 900 deliberately hard HardMatch pairs `fp8` costs 0.0074 mAA@10px, the
same as RoMaV2's own FP8 recipe.

**Sparse pipelines (RTX 5070 Ti `sm_120`):**

- **RaCo + ALIKED + LightGlue** (1024² pair, 1024 keypoints, exact ranker, `bf16` + `engine="compile"`): 14.6 ms per pair, against 18.1 ms for the TensorRT-FP16 engine of LightGlue-ONNX.
- **RDD** (1056x1600 extract): 110.9 ms in `fp32` eager, 35.4 ms with `fp8` + `engine="compile"`; the reference implementation without its CUDA extension runs out of memory on 16 GB.

## Configuring a run

The four presets are named partial configs; config files, dicts, dotted overrides and `load`'s keyword arguments (`device`,
`engine`, `strict`, `calibration`) layer on top.

```python
model = flashim.load("romav2", "fp8", overrides=["engine.kind=compile-full"])
model = flashim.load("raco-aliked-lightglue", "int8", config="run.toml")
model = flashim.load("aliked-lightglue", "int8", overrides={"components.extractor.preset": "fp32"})
flashim.resolve("rdd-lightglue", "fp8").to_dict()      # the config a build would use, without building
model.run_config.digest(), model.build_report          # what was asked for / what was actually applied
```

Unknown keys, passes and engines are errors; a pass the platform cannot run is skipped with a warning (`optimize.strict=true` raises instead). See
[docs/configuration.md](https://github.com/lnexenl/FlashIM/blob/main/docs/configuration.md).

## Platforms

| arch | GPU | status |
|---|---|---|
| sm_120 | RTX 5070 Ti | primary target: tuned and benchmarked |
| sm_90 | H20 | tuned; not re-tuned on H100 / H800 |
| sm_89 | L20 | tuned |
| sm_87 | Jetson AGX Orin | tuned (FP8 presets fall back to INT8) |
| sm_100 | datacenter Blackwell | compile-checked only |
| sm_80 / sm_86 | A100, RTX 30xx | not tuned; no FP8, so INT8 fallbacks apply |

A GPU without FP8 builds the nearest other preset instead (`int8`, or `bf16` for a model without INT8), and says so in a warning and in
`model.build_report.fallbacks`. Details, and which model was tested on which GPU (✅ / ❌ / 🔍), in
[docs/platforms.md](https://github.com/lnexenl/FlashIM/blob/main/docs/platforms.md).

## Checkpoints

Every model downloads its upstream release on first use into the torch hub checkpoint folder (`$TORCH_HOME/hub/checkpoints`) and verifies it
against the SHA-256 pinned in `flashim/models/<model>/weights.py`. Override `checkpoint` or set `FLASHIM_<MODEL>_CHECKPOINT` to use a local file;
copy the folder over for an offline machine. Calibration tables ship in the repository. More in [docs/checkpoints.md](https://github.com/lnexenl/FlashIM/blob/main/docs/checkpoints.md).

## Development

```bash
uv sync --extra eval                     # Python 3.12, torch 2.11 (cu130), triton 3.6
uv run pytest -m "not slow"              # unit / ops / quant
uv run pytest tests/parity               # bit-exactness vs the vendored upstream code (needs weights + GPU)
uv run python benchmarks/speed/forward.py --preset fp8 --engine compile
uv run python benchmarks/accuracy/run.py --dataset mega1500 --preset fp8 --seed 17
```

Agents and contributors: start with [AGENTS.md](https://github.com/lnexenl/FlashIM/blob/main/AGENTS.md) (layering rules, how to add an op, backend, model or platform, and how changes are verified).
The layout and op dispatch are described in [docs/architecture.md](https://github.com/lnexenl/FlashIM/blob/main/docs/architecture.md).

## License

FlashIM is released under the Apache License 2.0 ([LICENSE](https://github.com/lnexenl/FlashIM/blob/main/LICENSE)), with two exceptions inside the package:

- code ported from [RoMaV2](https://github.com/Parskatt/RoMaV2) (the RoMaV2 model and several Triton kernels) is under
  RoMaV2's MIT license ([`licenses/RoMaV2-MIT.txt`](https://github.com/lnexenl/FlashIM/blob/main/licenses/RoMaV2-MIT.txt));
- `flashim/models/romav2/dinov3.py` and `dpt.py` follow DINOv3 / DPT code by Meta that RoMaV2 includes, under the
  DINOv3 License Agreement ([`licenses/DINOv3-License.md`](https://github.com/lnexenl/FlashIM/blob/main/licenses/DINOv3-License.md)).

The package's license expression is therefore `Apache-2.0 AND MIT AND LicenseRef-DINOv3`. Model weights are downloaded
from their publishers, never shipped, and keep their upstream licenses, some more restrictive than Apache-2.0: the RoMaV2
checkpoint contains DINOv3 weights (DINOv3 License), SuperPoint (`superpoint_v1.pth`; `sp_v6` is MIT),
SuperGlue (Magic Leap academic/non-profit non-commercial research only) and RIPE are for non-commercial use only,
MASt3R, DUSt3R and UFM weights are CC BY-NC-SA 4.0 (non-commercial use only); RoMa (v1) is MIT with frozen DINOv2
weights (Apache-2.0); MINIMA (`minima-loftr`, `minima-roma`) is Apache-2.0 (with frozen DINOv2 Apache-2.0 for
`minima-roma`); LoMa is MIT except its LightGlue-derived matcher (Apache-2.0), and its checkpoints hold DINOv2
(Apache-2.0); OmniGlue and DINOv2 ViT-B/14 are Apache-2.0; EfficientLoFTR's mirror weights (`vismatch/eloftr`) and
vendored test oracle (`ffd4a46440`) are Apache-2.0, while upstream `zju3dv/EfficientLoFTR` re-licensed to the
ZJU3DV Project Registration License (PRL) v1.0 on 2026-09-15 (commit `07e9c1401e`). Details, and
the licenses of vendored test code and assets: [THIRD_PARTY_NOTICES.md](https://github.com/lnexenl/FlashIM/blob/main/THIRD_PARTY_NOTICES.md).

<p align="center">
  <img alt="" src="https://raw.githubusercontent.com/lnexenl/FlashIM/main/assets/brand/mark-pair.svg" width="64" height="64">&nbsp;&nbsp;
  <img alt="" src="https://raw.githubusercontent.com/lnexenl/FlashIM/main/assets/brand/mark-match-bolt.svg" width="64" height="64">&nbsp;&nbsp;
  <img alt="" src="https://raw.githubusercontent.com/lnexenl/FlashIM/main/assets/brand/mark-monogram.svg" width="64" height="64">
</p>
