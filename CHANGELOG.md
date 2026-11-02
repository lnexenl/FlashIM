# Changelog

## Unreleased

### Performance

- DeDoDe `int8` (ported from the LoMa experiments): `dedode-g` quantizes its DINOv2 ViT with per-token dynamic INT8
  (`DynamicInt8Linear`: `ops.quantize_int8_rows` + `ops.int8_linear` with per-row scales) instead of leaving it in BF16,
  and both models fuse their ConvRefiner decoders (depthwise 5x5 + ReLU as the INT8 producer of the 1x1 conv,
  upsample-concat writing the INT8 refiner input, the residual's 1/1.4 folded into `out_conv`). Extract at 1200 px on
  an RTX 5070 Ti (eager): `dedode` 46.4 -> 33.9 ms, `dedode-g` 139.9 -> 87.1 ms (on par with its `fp8`); MegaDepth-1500
  AUC@5 vs `bf16` +0.0004 / +0.0025.
  `extractor_spec(int8_dynamic=...)` and the `INT8_TOKEN_ACTIVATIONS` scheme make per-token INT8 available to other
  models; `Op.supports(...)` tells whether a call would run a kernel rather than the reference.
- Decoder upsampling at the memory roofline on sm_89 / sm_90 (bit-identical): `ops.upsample_concat_int8` has a second
  Triton impl, `triton_rows` (one program per pair of output rows, constexpr channel counts, vectorized loads, shared taps
  and horizontal lerps), chosen for exact 2x upsampling of BF16 maps; `ops.upsample_add` has a row kernel for wide
  channels-last maps. Other ratios (the DINOv2 descriptor's 56 -> 98 map), FP32 taps, narrow or NCHW maps and other
  architectures keep the tile kernel. Per extract at 1200 px, 4096 keypoints (CUDA graph, DRAM-bound), `dedode`:
  L20 upsample-concat 2.97 -> 1.54 ms, upsample-add 1.12 -> 0.89 ms; H20 3.76 -> 0.44 ms and 0.29 -> 0.25 ms; int8 pair
  (2 x extract + match) L20 68.8 -> 66.4 ms, H20 62.0 -> 57.5 ms.

## 0.1.0 (2026-10-28)

First public release.

### Interface

- One loader: `flashim.load(name, preset, *, device, engine, strict, calibration, config, overrides)`. Model-specific
  settings are config keys (`overrides={"arch.setting": "mega1500"}`); the per-model `load_*` functions are gone.
- One matching call: `model.match(a, b, *, resize, max_matches, threshold)` returns `flashim.Matches` for every model
  (dense, detector-free, detector + descriptor, LightGlue pipelines). Dense models attach the full prediction as
  `Matches.dense`; RoMaV2's `match_dense` returns only that. Per-call defaults come from the config's `inference`
  section.
- Four presets: `fp32`, `bf16`, `int8`, `fp8`. A model without the asked preset, and `fp8` on a GPU without FP8,
  builds the nearest one (`fp8` -> `int8` -> `bf16` -> `fp32`) and records it in `model.build_report.fallbacks`. A
  pipeline's preset applies to each of its components.
- Keypoints of every model are `(x, y)` pixel-center coordinates of the original image, RaCo's included (RaCo's own
  pixel-edge convention, pixel index + 0.5, is only what LightGlue sees in the `raco-*-lightglue` pipelines).
- `flashim.resolve` returns the config a build would use; `model.run_config` and `model.build_report` say what was
  asked for and what was applied.

### Models

- Dense: RoMa v2 (`bf16`, `int8`, `fp8`; `compile-full` engine).
- Sparse: RDD, ALIKED, RaCo (+ ALIKED or any descriptor model), SuperPoint, ALIKE, DISK, XFeat, DeDoDe(-G), DaD,
  RIPE, SIFT, ORB; semi-dense LoFTR; LightGlue pipelines for RDD, ALIKED, RaCo + ALIKED, SuperPoint, DISK and XFeat.
- Every learned model is checked against its vendored upstream code (`tests/parity`), and checkpoints download from
  their upstream releases, pinned by SHA-256.

### Packaging

- Python 3.10 or newer (3.10 for JetPack 6's torch); torch 2.11.x, torchvision 0.26.x, Triton 3.6.x.
- Extras: `cv` (OpenCV, for SIFT / ORB), `eval` (accuracy benchmarks), `tilelang`. A model that needs a missing
  extra says which one at load time.
- The wheel ships the calibration tables, the CUDA C++ sources and the third-party license texts.

### Performance

- MXFP8 and INT8 quantization with shipped calibration tables, fused Triton / TileLang / CUDA C++ kernels and
  `torch.compile`; accuracy accepted by pose AUC on MegaDepth-1500 / ScanNet-1500 and HardMatch
  ([docs/models.md](docs/models.md), [docs/benchmarks.md](docs/benchmarks.md)).
