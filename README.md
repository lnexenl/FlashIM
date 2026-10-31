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

FlashIM runs image-matching models (dense RoMa v2; sparse RDD, ALIKED, RaCo, SuperPoint, DISK, DeDoDe, LoFTR and others; LightGlue and LoMa matching) faster than their reference
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
```

`match(a, b, *, resize, max_matches, threshold)` is the same for every model: `resize` is the long side of the working
resolution, `max_matches` bounds the matches the model's own way (keypoints per image, the most confident matches, or
samples of a dense warp) and `threshold` is the model's match threshold. A model without the preset asked for builds the
nearest one it has (`fp8` -> `int8` -> `bf16` -> `fp32`), and a pipeline's preset applies to each of its components.

Weights download on first use and are checked against pinned SHA-256 hashes ([Checkpoints](https://github.com/lnexenl/FlashIM#checkpoints)).

## Results

All numbers are from an RTX 5070 Ti. The latency of every model, preset and engine is in the
[inference benchmark](https://github.com/lnexenl/FlashIM/blob/main/docs/benchmarks.md); models, presets and accuracy evidence are in [docs/models.md](https://github.com/lnexenl/FlashIM/blob/main/docs/models.md) and the page of each model.

**RoMaV2** (dense; 800² + 1024², both directions, forward only, `compile-full`):

| preset | forward | vs `bf16` |
|---|---|---|
| `bf16` | 195 ms | 1.00x |
| `fp8` | 110 ms | 1.77x |
| `int8` | 104 ms | 1.87x |

Pose AUC@5 vs `bf16` (3 seeds, MegaDepth-1500 / ScanNet-1500): `fp8` -0.0023 / +0.0007, `int8` -0.0010 / +0.0004, inside the RANSAC seed noise of about 0.003.
On the 900 deliberately hard HardMatch pairs `fp8` costs 0.0074 mAA@10px, the same as RoMaV2's own FP8 recipe.

**Sparse pipelines:**

- **RaCo + ALIKED + LightGlue** (1024² pair, 1024 keypoints, exact ranker): 14.6 ms per pair, against 18.1 ms for the TensorRT-FP16 engine of LightGlue-ONNX.
- **RDD** (1056x1600 extract): 110.9 ms in FP32, 35.4 ms with `fp8` + `compile`; the reference implementation without its CUDA extension runs out of memory on 16 GB.

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
`model.build_report.fallbacks`. Details in
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
checkpoint contains DINOv3 weights (DINOv3 License), SuperPoint and RIPE are for non-commercial use only; LoMa is MIT
except its LightGlue-derived matcher (Apache-2.0), and its checkpoints hold DINOv2 (Apache-2.0). Details, and
the licenses of vendored test code and assets: [THIRD_PARTY_NOTICES.md](https://github.com/lnexenl/FlashIM/blob/main/THIRD_PARTY_NOTICES.md).

<p align="center">
  <img alt="" src="https://raw.githubusercontent.com/lnexenl/FlashIM/main/assets/brand/mark-pair.svg" width="64" height="64">&nbsp;&nbsp;
  <img alt="" src="https://raw.githubusercontent.com/lnexenl/FlashIM/main/assets/brand/mark-match-bolt.svg" width="64" height="64">&nbsp;&nbsp;
  <img alt="" src="https://raw.githubusercontent.com/lnexenl/FlashIM/main/assets/brand/mark-monogram.svg" width="64" height="64">
</p>
