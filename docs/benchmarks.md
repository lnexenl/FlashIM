# Inference benchmark

Latency of every model, preset and engine, measured on one GPU with one protocol. The tables are latency only: check
[Accuracy](#accuracy) before picking the fastest cell.

- **Setup**: NVIDIA GeForce RTX 5070 Ti (sm_120), torch 2.11.0+cu130, measured 2026-10-06.
- **Timing**: median of 50 calls timed with CUDA events, after 10 untimed warmups (compilation included). One
  configuration per process.
- **Input**: images are decoded and on the GPU before the timing starts. RoMaV2 pairs are preprocessed once; the sparse
  models resize inside the timed call.
- **Units**: milliseconds. The fastest number of a row is in bold.

Presets are described in [models.md](models.md).

## RoMaV2 (dense)

Preloaded forward in both directions, mega1500 setting (800² LR + 1024² HR), B=1, run with `compile-full`, best of two processes
(compiled runs vary 15-25% between processes). Speed-up over `bf16`.

| preset | forward | speed-up |
|---|---|---|
| `bf16` | 195.0 | 1.00x |
| `int8` | 104.0 | 1.87x |
| `fp8` | 110.0 | 1.77x |

## Sparse extractors

`extract` of one MegaDepth image at 1200 px on the long side, 4096 keypoints. Each cell is `eager / compile`; an empty
cell is a preset the model does not have.

| model | fp32 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| superpoint | 12.7 / 12.5 | 6.63 / 5.57 | 4.85 / **3.60** | 4.73 / 3.79 |
| disk | 14.2 / 11.2 | 8.35 / 7.27 | 9.79 / **7.04** | 7.82 / 7.07 |
| alike | 6.71 / 5.84 | 3.96 / **3.06** | 4.70 / 4.56 | 4.92 / 4.82 |
| xfeat | 2.22 / 2.19 | 2.27 / 1.86 | 2.72 / **1.81** | 2.62 / 2.04 |
| dedode | 197.9 / 125.0 | 56.9 / 61.9 | **33.9**¹ / 50.7 | 41.0 / 51.7 |
| dedode-g | 543.7 / 468.8 | 130.9 / 132.1 | 87.1¹ / 120.5 | **87.2** / 100.9 |
| dad | 520.2 / 420.1 | 113.2 / 120.9 | 102.4 / 108.5 | **75.1** / 88.7 |
| ripe | 114.4 / 72.5 | 33.6 / fail | **24.5** / fail | 25.1 / fail |
| aliked | 9.05 / 10.3 | 5.34 / **5.17** |  |  |
| raco-aliked | 29.6 / 22.3 | 9.18 / **8.42** |  |  |
| rdd | 69.0 / 61.4 | 34.8 / 25.6 |  | 29.4 / **22.5** |
| sift | **26.0** / - |  |  |  |
| orb | **10.4** / - |  |  |  |

- SIFT and ORB are OpenCV (fp32, eager only).
- ¹ Remeasured 2026-11-01 (same protocol) after the DeDoDe int8 decoder fusions and, for `dedode-g`, the per-token
  INT8 ViT (were 46.4 and 139.9). Their compile cells are the old run's: the int8 sparse extractors do not compile at
  the moment (Dynamo rejects the autotuned INT8 conv kernels).
- `ripe` with `engine="compile"` fails in Inductor for `bf16`, `int8` and `fp8` (`InductorError: KeyError`); `fp32`
  compiles.

## Pair matching

One image pair at 1200 px. Each cell is `eager / compile`.

- `*-lightglue`: 2 x `extract` + LightGlue `match_features` with 4096 keypoints per image. The column is LightGlue's
  preset. The extractor runs the same preset when it has one, else `bf16` (ALIKED and RaCo-ALIKED for int8 and fp8, RDD
  for int8). `match_features` is the median of 20 calls after 3 warmups.
- `loftr`: `match` of the pair, all coarse matches.

| pipeline | fp32 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| superpoint-lightglue | 80.9 / 79.9 | 27.6 / 22.1 | 20.2 / 19.7 | 23.3 / **15.9** |
| disk-lightglue | 86.4 / 79.8 | 31.1 / 26.1 | 26.4 / 26.5 | 27.1 / **25.2** |
| xfeat-lightglue | 21.2 / 20.8 | 8.39 / **7.14** | 10.1 / 7.42 | 11.0 / 7.88 |
| aliked-lightglue | 76.0 / 77.2 | 25.9 / 22.4 | 21.6 / **21.0** | 22.0 / 21.4 |
| raco-aliked-lightglue | 123.3 / 102.0 | 37.1 / 30.5 | 29.4 / 27.9 | 29.8 / **27.7** |
| rdd-lightglue | 214.6 / 221.4 | 97.1 / 62.7 | 80.6 / 72.1 | 70.5 / **56.0** |
| loftr | 132.6 / 131.7 | 68.4 / 58.1 | 45.3 / 48.5 | 47.7 / **42.5** |

*(The tables above are the 2026-10-06 compiled/eager baseline on RTX 5070 Ti `sm_120`; full `engine="compile"` / `compile-full` tables were not re-measured in the 2026-11 sweep. Since 2026-10-06, `dedode`, `aliked-lightglue`, `superpoint-lightglue`, and `disk-lightglue` omit `fp8` -> `int8`, and `xfeat-lightglue` omits `int8`/`fp8` -> `bf16`.)*

## Eager sm_120 speedups (2026-11 sweep)

Before/after eager timings measured during the 2026-11 `sm_120` optimization sweep on **NVIDIA GeForce RTX 5070 Ti (`sm_120`), torch 2.11.0+cu130, triton 3.6, `engine="eager"`** (CUDA-event / interleaved timings from each change; configurations not directly measured in this sweep — including all `engine="compile"` / `compile-full` rows and unmeasured `(model, preset)` cells — are omitted rather than estimated).

| model / stage | shape | preset | base (ms) | new (ms) | speed-up |
|---|---|---|---:|---:|---:|
| `romav2` forward (both dirs, B=1) | `mega1500` (800² + 1024²) | `fp8` | 184.07 | 113.94 | 1.62x |
| `romav2` forward (both dirs, B=1) | `mega1500` (800² + 1024²) | `int8` | 139.75 | 110.83 | 1.26x |
| `loftr` pair (diff shapes) | 896x1200 / 880x1200 | `bf16` / `int8` / `fp8` | 94.03 / 60.73 / 57.91 | 79.62 / 37.99 / 45.81 | 1.18x / 1.60x / 1.26x |
| `lightglue` (`rdd` matcher) | N = 1024 | `fp8` / `int8` | 7.09 / 6.29 | 1.56 / 1.48 | 4.55x / 4.25x |
| `lightglue` (`aliked` matcher) | N = 2048 | `fp8` / `int8` | 7.07 / 6.03 | 3.44 / 3.36 | 2.06x / 1.79x |
| `lightglue` (`rdd` matcher) | N = 4096 | `fp8` / `int8` | 13.01 / 12.60 | 10.24 / 9.90 | 1.27x / 1.27x |
| `loma-b` matcher | N = 2048 | `fp8` | 7.21 | 3.52 | 2.05x |
| `loma-l` matcher | N = 4096 | `int8` / `fp8` | 20.32 / 26.22 | 12.97 / 18.61 | 1.57x / 1.41x |
| `loma-g` matcher | N = 4096 | `int8` / `fp8` | 43.57 / 65.58 | 29.25 / 39.30 | 1.49x / 1.67x |
| `loma-b128` pair (2 ext + mat) | 1024 px, 4096 kp | `fp8` vs `bf16` | 51.17 (`bf16`) | 40.11 (`fp8`) | 1.28x |
| `orb` `match_features` | 4096 x 4096 | `fp32` | 34.061 | 0.312 | 109.1x |
| `sift` `match_features` | 4096 x 3629 | `fp32` | 1.344 | 0.849 | 1.58x |
| `superpoint` `extract` | 856x1200, 4096 kp | `bf16` / `int8` / `fp8` | 8.152 / 6.031 / 6.089 | 4.421 / 1.347 / 3.824 | 1.84x / 4.48x / 1.59x |
| `alike` `extract` | 832x1184, 4096 kp | `bf16` / `int8` / `fp8` | 7.173 / 7.331 / 7.589 | 2.085 / 1.782 / 1.997 | 3.44x / 4.11x / 3.80x |
| `aliked` `extract` | 832x1184, 4096 kp | `bf16` / `fp32` | 5.418 / 6.357 | 3.470 / 5.590 | 1.56x / 1.14x |
| `raco` `extract_keypoints` | 1024x1024 | `bf16` | 6.919 | 3.673 | 1.88x |
| `disk` `extract` | 1200 px, 4096 kp | `bf16` / `int8` / `fp8` | 12.730 / 12.981 / 15.404 | 6.668 / 7.465 / 5.751 | 1.91x / 1.74x / 2.68x |
| `xfeat` `extract` | 1200 px, 4096 kp | `bf16` / `int8` / `fp8` | 2.010 / 2.253 / 2.442 | 1.316 / 1.131 / 1.417 | 1.53x / 1.99x / 1.72x |
| `rdd` `extract` | 1600 px | `bf16` / `fp8` | 62.72 / 53.59 | 56.92 / 47.76 | 1.10x / 1.12x |
| `rdd` `match_features` | 4096 x 4096 | `bf16` | 5.76 | 0.75 | 7.67x |
| `dedode` `extract` | 1200 px, 4096 kp | `int8` vs `bf16` | 55.94 (`bf16`) | 30.37 (`int8`) | 1.84x |
| `dedode-g` `extract` | 1200 px | `int8` / `fp8` | 101.55 / 116.36 | 76.14 / 101.22 | 1.33x / 1.15x |
| `dad` `extract` | 1200 px | `int8` / `fp8` | 136.5 / 102.79 | 78.47 / 85.97 | 1.74x / 1.20x |
| `ripe` `extract` | 1200 px | `int8` | 42.28 | 19.19 | 2.20x |
| `ripe` `match_features` | 32320 x 23731 x 256 | `fp32` (TF32 > 8192²) | 22.54 | 9.81 | 2.30x (-2.97 GB VRAM) |

## Accuracy

### Accuracy of the pre-2026-11 models (2026-11 gate, sm_120)

All 26 pre-2026-11 models evaluated on **MegaDepth-1500** and **ScanNet-1500** (1500 pairs each) on **RTX 5070 Ti (`sm_120`)**, PoseLib LO-RANSAC at 1.0 px, 3 seeds `(17, 29, 42)`, `FLASHIM_IMAGE_DECODER=pil`, paired bootstrap 95% CI (`B=10000`) against the `dev` `bf16` baseline (`fp32` for `orb` and `sift`; `engine="compile"` for `romav2`, `engine="eager"` for all others; working resolution per `benchmarks/accuracy/perf_protocol.py`, e.g. 1200 / 640 px and 4096 kp for classic extractors and LightGlue, 1600 px for `rdd`, 1024 px / 4096 kp for `loma-*`, 800² + 1024² for `romav2`).

Gate: an AUC@5 drop below 0.005, decided with a 95% paired bootstrap CI over pairs. **84 of the 126 active `(model, preset, dataset)` cells pass, 42 are borderline (`†`: CI straddles -0.005), and 0 fail.** `→ int8` / `→ bf16` mark presets omitted and resolved via `build.NEAREST`; `—` means the model does not define that preset.

| model | MegaDepth base AUC@5 | bf16 Δ | int8 Δ | fp8 Δ | ScanNet base AUC@5 | bf16 Δ | int8 Δ | fp8 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `orb` (`fp32`) | 0.2011 | +0.0000 | — | — | 0.0612 | +0.0000 | — | — |
| `sift` (`fp32`) | 0.4758 | +0.0000 | — | — | 0.0848 | +0.0000 | — | — |
| `superpoint` | 0.4529 | -0.0007† | -0.0010† | -0.0027† | 0.1382 | +0.0000 | -0.0003† | +0.0009† |
| `xfeat` | 0.4385 | +0.0038 | -0.0002† | +0.0027 | 0.0936 | +0.0022 | +0.0100 | +0.0025 |
| `alike` | 0.5626 | +0.0020 | +0.0041 | +0.0002† | 0.1359 | +0.0004 | +0.0031 | +0.0079 |
| `aliked` | 0.4631 | +0.0559 | — | — | 0.0724 | +0.0068 | — | — |
| `disk` | 0.5489 | +0.0031 | +0.0029 | -0.0011† | 0.1275 | +0.0007 | +0.0017 | -0.0005† |
| `raco-aliked` | 0.5234 | +0.0397 | — | — | 0.0691 | +0.0056 | — | — |
| `rdd` | 0.6433 | -0.0003 | — | -0.0004 | 0.1705 | +0.0035 | — | +0.0025 |
| `dedode` | 0.6164 | +0.0000 | -0.0022† | → int8 | 0.1620 | +0.0000 | +0.0011 | → int8 |
| `dedode-g` | 0.6490 | +0.0000 | -0.0010† | -0.0033† | 0.2004 | +0.0000 | +0.0035 | +0.0002† |
| `dad` | 0.6491 | +0.0000 | +0.0013 | +0.0006† | 0.1987 | +0.0000 | +0.0025 | -0.0007† |
| `ripe` | 0.5764 | +0.0000 | -0.0016† | -0.0020† | 0.1507 | -0.0000 | -0.0002† | -0.0005† |
| `superpoint-lightglue` | 0.5971 | +0.0010 | +0.0009† | → int8 | 0.1799 | +0.0000 | +0.0033 | → int8 |
| `xfeat-lightglue` | 0.5580 | +0.0043 | → bf16 | → bf16 | 0.1580 | -0.0001 | → bf16 | → bf16 |
| `disk-lightglue` | 0.6126 | +0.0036 | +0.0039 | → int8 | 0.1829 | -0.0026† | +0.0010† | → int8 |
| `aliked-lightglue` | 0.6112 | +0.0126 | +0.0103 | → int8 | 0.2015 | +0.0073 | +0.0089 | → int8 |
| `raco-aliked-lightglue` | 0.6205 | +0.0186 | +0.0201 | +0.0169 | 0.1975 | +0.0122 | +0.0109 | +0.0080 |
| `rdd-lightglue` | 0.6547 | +0.0015 | +0.0017 | -0.0045† | 0.2126 | -0.0000 | +0.0002† | +0.0014† |
| `loftr` | 0.6648 | -0.0005 | -0.0018 | -0.0019† | 0.2181 | +0.0018 | +0.0002 | -0.0018† |
| `loma-b128` | 0.6756 | -0.0016¹ | +0.0038 | +0.0010 | 0.2980 | -0.0035†¹ | -0.0013† | -0.0008† |
| `loma-b` | 0.6746 | +0.0015 | +0.0078 | +0.0019 | 0.3136 | -0.0009 | -0.0011† | -0.0008† |
| `loma-r` | 0.6746 | -0.0026† | +0.0008 | +0.0010 | 0.3023 | +0.0011 | -0.0001† | +0.0008† |
| `loma-l` | 0.6821 | -0.0012 | +0.0011 | -0.0022† | 0.3250 | -0.0013 | -0.0016† | -0.0006† |
| `loma-g` | 0.6789 | +0.0000 | +0.0031 | +0.0008 | 0.3282 | +0.0000 | -0.0018† | -0.0019† |
| `romav2` (`compile`) | 0.6231 | -0.0007 | -0.0032† | -0.0044† | 0.3374 | -0.0006 | +0.0010 | -0.0027 |

- ¹ `loma-b128 bf16` was subsequently re-verified bit-exact (`+0.0000` on both MegaDepth-1500 and ScanNet-1500) after unbatching `CrossBlock` FFN.
- **ALIKED family gain**: `aliked`, `raco-aliked`, `aliked-lightglue`, and `raco-aliked-lightglue` score higher than `dev bf16` because SDDH pixel-space keypoint coordinates now stay in FP32 instead of quantizing to BF16 (2–8 px ULP).

### Historical 2026-10-05 MegaDepth-1500 baseline (before 2026-11 fixes)

A faster cell is not always a usable one. Before the 2026-11 quantization fixes and preset fallbacks above, this was the
pose AUC@5 change of each quantized preset against the same model's `bf16` on MegaDepth-1500 (4 RANSAC seeds, `eager`,
RTX 5070 Ti `sm_120`, run 2026-10-05; cells that failed the -0.005 gate are in bold):

| model | int8 | fp8 |
|---|---|---|
| superpoint | -0.0009 | +0.0052 |
| disk | +0.0018 | -0.0002 |
| alike | +0.0074 | +0.0020 |
| xfeat | **-0.0073** | **-0.0109** |
| dedode | +0.0023 | **-0.0069** |
| dedode-g | +0.0014 | +0.0005 |
| dad | -0.0019 | -0.0003 |
| ripe | -0.0016 | -0.0040 |
| loftr | +0.0001 | -0.0025 |
| superpoint-lightglue | **-0.0085** | **-0.0226** |
| disk-lightglue | +0.0011 | -0.0022 |
| xfeat-lightglue | **-0.0244** | **-0.1366** |

`dedode` / `dedode-g` int8 after the 2026-11-01 changes (decoder fusions, per-token INT8 ViT), 3 seeds against a `bf16`
run of the same tree: +0.0004 / +0.0025.

## Accuracy of the 2026-11 additions

Pose AUC@5 of `roma`, `tiny-roma`, `minima-roma`, `mast3r`, `dust3r`, `ufm`, `eloftr`, `xoftr`, `minima-loftr`,
`superpoint-superglue` and `omniglue`, on MegaDepth-1500 and ScanNet-1500 (1500 pairs each, RTX 5070 Ti `sm_120`, `eager`). Each cell is the
change of the quantized preset against the same model's `bf16`; a model's presets share one protocol (its native
resolution, e.g. `roma` 560 -> 864, `mast3r` / `dust3r` 512, `eloftr` / `minima-loftr` 840 on MegaDepth, 640 x 480 on
ScanNet, `superpoint-superglue` / `omniglue` 1024 px with 1024 keypoints on MegaDepth; dense models keep 5000 sampled
matches; poselib LO-RANSAC at 1.0 px, seeds 17, 29, 42 on the same matches). `roma` is the outdoor model on
MegaDepth and `roma-indoor` on ScanNet; `superpoint-superglue` uses its indoor matcher on ScanNet.

Gate: an AUC@5 drop below 0.005, decided with a 95% paired bootstrap CI over pairs. **27 of the 44 quantized cells
pass, 17 are inconclusive (†: the CI straddles -0.005; a 1500-pair AUC@5 delta resolves about +-0.005), none fails.**
`fp8` of `superpoint-superglue` and `omniglue` is not a preset (-> `int8`): their fp8 extractors lost 0.013-0.014 AUC@5
(SuperPoint `conv1b` has 41.6x block-32 channel outliers) and 0.005-0.006 (DINOv2 ViT-B/14 under MXFP8) against `int8`.
`dust3r`'s `bf16` (FP16 compute) is itself +0.0096 above its `fp32` on MegaDepth, so its quantized cells are within
+0.003 of `fp32`.

| model | MegaDepth bf16 AUC@5 | int8 Δ | fp8 Δ | ScanNet bf16 AUC@5 | int8 Δ | fp8 Δ |
|---|---:|---:|---:|---:|---:|---:|
| `roma / roma-indoor` | 0.6931 | -0.0010 | +0.0005 | 0.3340 | +0.0010 | -0.0002 |
| `tiny-roma` | 0.5411 | -0.0031† | -0.0030† | 0.1482 | -0.0020† | -0.0008† |
| `minima-roma` | 0.6902 | +0.0002 | +0.0008 | 0.3065 | +0.0008 | +0.0000† |
| `mast3r` | 0.5912 | -0.0016† | +0.0001 | 0.3620 | -0.0001 | -0.0004 |
| `dust3r` | 0.3555 | -0.0069† | -0.0083† | 0.3009 | +0.0009 | -0.0016 |
| `ufm` | 0.3635 | +0.0026 | -0.0027† | 0.3219 | -0.0028† | -0.0006 |
| `eloftr` | 0.6523 | +0.0003 | +0.0019 | 0.2440 | +0.0054 | +0.0035 |
| `xoftr` | 0.6120 | +0.0010 | +0.0023 | 0.1854 | +0.0015 | +0.0025 |
| `minima-loftr` | 0.5518 | +0.0009 | +0.0014 | 0.1339 | -0.0005 | -0.0026† |
| `superpoint-superglue` | 0.5416 | -0.0028† | → int8 | 0.1717 | +0.0068 | → int8 |
| `omniglue` | 0.5195 | +0.0024 | → int8 | 0.1748 | +0.0012† | → int8 |

Latency of these models is not in the tables above (they were added after that run). Per-model eager timings measured
while porting are in each model's `quant.py` docstring.

## Running it

The full table takes about 3.5 hours on an RTX 5070 Ti:

```bash
mkdir -p results/speed/table
setsid nohup benchmarks/speed/speed_table.sh >> results/speed/table/run.log 2>&1 &
uv run python benchmarks/speed/speed_table.py          # writes results/speed/table/table.md
```

`speed_table.sh` is resumable: a configuration that has a result is skipped, and a failed one is retried.

A single configuration:

```bash
# RoMaV2
uv run python benchmarks/speed/forward.py --preset fp8 --engine compile-full

# a sparse extractor
uv run python benchmarks/speed/extractor.py --model superpoint --preset int8 --engine compile --resize 1200

# a pipeline: extract + match
uv run python benchmarks/speed/extractor.py --model disk-lightglue --preset int8 --resize 1200 \
    --overrides components.extractor.preset=int8
```

The accuracy matrix of the 2026-11 additions (about 6.7 hours on an RTX 5070 Ti; resumable, each run acquires a GPU lock
so it can share the GPU) and its report with the paired-bootstrap gate:

```bash
benchmarks/accuracy/b_models_matrix.sh > results/accuracy/b_models/matrix.log 2>&1
uv run python benchmarks/accuracy/b_models_report.py --dir results/accuracy/b_models
```

The accuracy matrix of the pre-2026-11 models against a base checkout (resumable, supports `OUT_DIR`, `MAX_PAIRS`,
positional model filters, and `SKIP_SPECS="model:preset ..."` to skip unchanged entries when re-running) and its paired
bootstrap report:

```bash
OUT_DIR=results/accuracy/perf benchmarks/accuracy/perf_matrix.sh > results/accuracy/perf/matrix.log 2>&1
SKIP_SPECS="orb:fp32 sift:fp32 rdd:bf16 rdd:fp8" OUT_DIR=results/accuracy/perf3 \
    benchmarks/accuracy/perf_matrix.sh dedode ripe xfeat-lightglue
uv run python benchmarks/accuracy/perf_report.py --dir results/accuracy/perf3
```

`benchmarks/accuracy/run_sparse.py --dataset scannet1500` is the generic ScanNet-1500 runner (any model that returns
`Matches`; `run.py` is RoMaV2's).
