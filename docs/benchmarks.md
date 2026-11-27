# Inference benchmark

Latency and accuracy of the current code, per model family: [dense and semi-dense models](#dense-and-semi-dense-models)
(one pair), [sparse extractors](#sparse-extractors) (one image), [sparse matchers](#sparse-matchers) (two sets of
keypoints), the [pair time of sparse pipelines](#pair-time-of-sparse-pipelines) (two extractions plus the match) and
[FlexGlue](#flexglue-sparse-to-dense). Check [Accuracy](#accuracy) before picking the fastest cell.

- **Setup**: NVIDIA GeForce RTX 5070 Ti (`sm_120`), torch 2.11.0+cu130, triton 3.6.
- **Timing**: median of 50 calls timed with CUDA events, after 10 untimed warmups (compilation included), one
  configuration per process. Compiled RoMaV2 runs are the best of two processes (they vary 15-25% between processes).
- **Input**: images are decoded and on the GPU before the timing starts. The sparse models resize inside the timed call.
- **Cells**: milliseconds, `eager / compile`; the fastest number of a row is in bold. `→ int8` / `→ bf16`: the model has no
  such preset and builds the one shown (see [models.md](models.md)); an empty cell is a preset the model does not have;
  `fail`: the run crashed ([Compile failures](#compile-failures)); `-`: not measured.

Presets are described in [models.md](models.md).

## Dense and semi-dense models

RoMaV2: preloaded bidirectional forward, mega1500 setting (800² LR + 1024² HR), B=1. Compiled engines: best of two processes. Speed-up: over `bf16` in the same engine.

| preset | eager | compile | compile-full | speed-up (compile-full) |
|---|---|---|---|---|
| `bf16` | 348.5 | 335.4 | **194.3** | 1.00x |
| `int8` | 107.1 | 106.7 | **93.4** | 2.08x |
| `fp8` | 110.7 | 119.6 | **106.5** | 1.82x |

`match` of one MegaDepth pair at each model's own default resolution; dense models keep 5000 sampled matches (`max_matches=5000`), semi-dense ones all their matches. Each cell is `eager / compile`.

| model | fp32 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| loftr | 123.1 / 112.4 | 56.6 / 45.8 | 54.3 / 54.1 | 35.2 / **34.9** |
| roma | 444.9 / 397.6 | 122.7 / 111.1 | 63.8 / **63.8** | 85.6 / 80.1 |
| minima-roma | 445.0 / 397.5 | 122.8 / 111.2 | **63.8** / 63.8 | 85.6 / 79.9 |
| tiny-roma | 20.9 / 20.7 | 8.29 / 6.15 | 7.23 / **5.35** | 7.46 / 5.56 |
| mast3r | 108.1 / 103.5 | 28.7 / 26.7 | 18.9 / 21.7 | **18.6** / 25.0 |
| dust3r | 103.4 / 97.3 | 30.6 / 28.7 | 20.6 / 23.4 | **20.4** / 26.4 |
| ufm | 146.8 / 141.1 | 40.0 / 37.9 | **26.2** / 29.2 | 26.3 / 28.7 |
| eloftr | 143.4 / 139.6 | 52.9 / 65.7 | **38.0** / 51.1 | 41.3 / 54.1 |
| xoftr | 23.5 / 22.5 | 10.3 / 9.48 | **7.55** / fail | 7.74 / fail |
| minima-loftr | 240.2 / 221.2 | 103.9 / **98.4** | 104.7 / 104.8 | 99.9 / 100.0 |

## Sparse extractors

`extract` of one MegaDepth image, 1200 px long side, 4096 keypoints (1024 px / 1024 keypoints for `omniglue` and `superpoint-superglue`, 1024 px / 4096 for `loma-*`); each cell is `eager / compile`. SIFT and ORB are OpenCV (fp32, eager).

| model | fp32 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| superpoint | 12.4 / 12.1 | 4.78 / 4.45 | 1.49 / **1.49** | 3.76 / 3.66 |
| disk | 13.8 / 11.0 | 6.41 / 6.11 | **5.14** / 5.39 | 5.27 / 5.48 |
| alike | 4.03 / 3.15 | 2.21 / **1.57** | 1.77 / fail | 1.93 / fail |
| xfeat | 1.56 / 1.52 | 1.56 / 1.50 | 1.43 / fail | **1.42** / fail |
| dedode | 174.6 / 115.2 | 54.4 / 50.9 | **30.2** / fail | → int8 / fail |
| dedode-g | 532.5 / 460.6 | 128.2 / 121.0 | **65.7** / fail | 68.4 / fail |
| dad | 462.2 / 416.4 | 111.8 / 106.7 | **49.8** / fail | 57.4 / fail |
| ripe | 110.2 / 72.4 | 31.4 / fail | **18.8** / fail | 25.5 / fail |
| ripepp | 109.6 / 76.4 | 32.9 / fail | **20.5** / fail | 26.6 / fail |
| aliked | 5.94 / 5.10 | 3.72 / **2.70** |  |  |
| raco-aliked | 26.8 / 18.1 | 7.03 / **5.76** |  |  |
| rdd | 68.7 / 61.0 | 31.8 / 25.3 |  | 26.5 / **22.8** |
| traqpoint | 90.5 / 79.5 | 37.5 / 30.8 | 33.6 / 29.0 | 32.2 / **28.3** |
| sift | **25.4** / - |  |  |  |
| orb | **10.3** / - |  |  |  |
| loma-b128 | 60.8 / 43.2 | 22.4 / 21.4 | **10.5** / fail | 17.3 / fail |
| loma-b | 224.2 / 201.5 | 59.2 / 56.6 | **26.3** / fail | 31.6 / fail |
| loma-r | 225.0 / 201.9 | 59.2 / 56.7 | **26.3** / fail | 31.6 / fail |
| loma-l | 225.1 / 202.1 | 59.2 / 56.6 | **27.7** / fail | 31.6 / fail |
| loma-g | 225.3 / 201.8 | 59.2 / 56.7 | **26.3** / fail | 31.6 / fail |
| superpoint-superglue | 8.80 / 8.71 | 3.43 / 3.23 | **1.15** / 1.17 |  |
| omniglue | 30.4 / 29.5 | 9.93 / 9.64 | **8.07** / 8.69 |  |

## Sparse matchers

`match_features` of two sets of keypoints with descriptors (the model's own matcher: LightGlue, LoMa, SuperGlue, OmniGlue; mutual nearest neighbours for `rdd`, `orb`, `sift`); median of 20 calls after 3 warmups. Each cell is `eager / compile`.

| model | fp32 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| superpoint-lightglue | 70.2 / 54.5 | 11.8 / 10.8 | **10.1** / 10.1 | → int8 / → int8 |
| disk-lightglue | 57.7 / 57.3 | 11.7 / 11.2 | **9.77** / 9.99 | → int8 / → int8 |
| xfeat-lightglue | 16.6 / 16.3 | 3.52 / **3.30** | → bf16 / → bf16 | → bf16 / → bf16 |
| aliked-lightglue | 57.6 / 57.2 | 11.7 / 11.2 | **9.78** / 10.0 | → int8 / → int8 |
| raco-aliked-lightglue | 57.7 / 57.4 | 11.7 / 11.1 | **9.81** / 10.0 | 10.2 / 17.8 |
| rdd-lightglue | 57.9 / 57.4 | 11.7 / 13.5 | **9.89** / 10.1 | 10.3 / 17.9 |
| ripepp-lightglue | 57.9 / 57.4 | **11.7** / - | 11.8 / - | → bf16 / - |
| loma-b128 | 57.6 / 57.2 | 11.7 / 11.0 | **9.77** / - | 10.5 / - |
| loma-b | 57.6 / 57.1 | 11.7 / 11.0 | **9.81** / - | 10.2 / - |
| loma-r | 57.9 / 57.2 | 11.7 / 11.0 | **9.82** / - | 10.3 / - |
| loma-l | 124.6 / 123.2 | 26.3 / 24.7 | **13.0** / - | 14.2 / - |
| loma-g | 315.0 / 308.2 | 66.7 / 62.5 | **29.2** / - | 32.2 / - |
| superpoint-superglue | 6.72 / 6.81 | 2.81 / 2.92 | **1.81** / 4.21 |  |
| omniglue | 7.57 / 7.50 | 3.96 / 3.96 | **3.04** / 6.53 |  |
| rdd | 0.75 / 0.76 | 0.75 / **0.74** |  | 0.75 / 0.75 |
| traqpoint | 0.75 / 0.76 | 0.75 / 0.75 | **0.74** / 0.75 | 0.75 / 0.75 |
| orb | **0.29** / - |  |  |  |
| sift | **0.89** / - |  |  |  |

## Pair time of sparse pipelines

One image pair: 2 x `extract` + the matcher's `match_features`, the column being the pipeline's preset, built as `flashim.load` builds it (each component gets the preset its spec gives it; the table below the grid lists the pipelines whose components differ). Same sizes as above. Each cell is `eager / compile`.

| pipeline | fp32 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| superpoint-lightglue | 99.5 / 78.5 | 21.3 / 19.7 | 19.6 / **19.0** | → int8 / → int8 |
| disk-lightglue | 85.4 / 79.2 | 24.5 / 23.0 | **20.0** / 20.3 | → int8 / → int8 |
| xfeat-lightglue | 19.7 / 19.4 | 6.59 / **6.27** | → bf16 / → bf16 | → bf16 / → bf16 |
| aliked-lightglue | 69.5 / 67.4 | 18.9 / 16.7 | 17.1 / **15.4** | → int8 / → int8 |
| raco-aliked-lightglue | 111.6 / 93.6 | 25.5 / 22.6 | 23.6 / **21.5** | 24.0 / 29.4 |
| rdd-lightglue | 195.5 / 179.5 | 75.2 / 64.1 | 73.4 / 64.6 | **63.2** / 63.9 |
| ripepp-lightglue | 277.4 / 201.5 | 72.9 / fail | **52.2** / fail | → bf16 / fail |
| loma-b128 | 179.3 / 143.6 | 56.5 / 53.9 | **30.8** / fail | 45.0 / fail |
| loma-b | 506.1 / 460.1 | 130.1 / 124.2 | **62.4** / fail | 73.5 / fail |
| loma-r | 507.8 / 461.0 | 130.1 / 124.5 | **62.4** / fail | 73.5 / fail |
| loma-l | 574.8 / 527.4 | 144.7 / 137.9 | **68.3** / fail | 77.4 / fail |
| loma-g | 765.6 / 711.8 | 185.2 / 175.9 | **81.8** / fail | 95.4 / fail |
| superpoint-superglue | 24.3 / 24.2 | 9.67 / 9.38 | **4.10** / 6.55 |  |
| omniglue | 68.3 / 66.6 | 23.8 / 23.2 | **19.2** / 23.9 |  |

Pipelines whose two components build different presets (`extractor+matcher`; a component without the requested preset builds the nearest one it has):

| pipeline | requested | built |
|---|---|---|
| aliked-lightglue | int8 | bf16+int8 |
| raco-aliked-lightglue | fp8 | bf16+fp8 |
| raco-aliked-lightglue | int8 | bf16+int8 |
| rdd-lightglue | int8 | bf16+int8 |
| ripepp-lightglue | int8 | int8+bf16 |
| superpoint-lightglue | int8 | bf16+int8 |

## FlexGlue (sparse-to-dense)

One MegaDepth pair, 512x512 descriptor: `match_keypoints` with 2048 RDD keypoints per image and dense `match()` with 5000 sampled matches. Each cell is `eager / compile`.

| call | fp32 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| `match_keypoints` | 145.1 / 143.6 | 35.1 / 33.7 | 22.3 / 22.2 | 18.0 / **17.9** |
| dense `match()` | 146.3 / 145.2 | 36.1 / 34.9 | 23.4 / 23.4 | **19.0** / 19.1 |

## Compile failures

`engine="compile"` crashes for these configurations; `eager` runs them all.

- **INT8 / FP8 sparse extractors, LoMa and RIPE++** (`alike`, `xfeat`, `dedode`, `dedode-g`, `dad`, `ripe`, `ripepp`,
  `loma-*`, `xoftr`; `int8` and `fp8`, and `ripepp-lightglue` `int8`): Dynamo rejects the autotuned INT8 convolution kernels
  (`AssertionError: ListVariable ... is already tracked for mutation`, torch 2.11); they run eager.
- **`ripe`, `ripepp`, `ripepp-lightglue`** (`bf16`; `fp32` compiles): `InductorError: KeyError` in the Inductor scheduler
  (torch 2.11).

## Accuracy

Pose AUC@5 on MegaDepth-1500 and ScanNet-1500 (1500 pairs each), RTX 5070 Ti (`sm_120`), PoseLib LO-RANSAC at 1.0 px, 3
seeds (17, 29, 42), paired bootstrap 95% CI (`B=10000`). Gate: an AUC@5 drop below 0.005, decided with the CI over pairs
(**PASS** if the CI's lower bound is above -0.005, **FAIL** if its upper bound is below, **borderline** `†` if it
straddles -0.005; a 1500-pair AUC@5 delta resolves about +-0.005). `→ int8` / `→ bf16` mark presets omitted and resolved
via `build.NEAREST`; `—` means the model does not define that preset. Tables are grouped like the latency above; within
a family, the first table compares against the `bf16` of the unoptimized code and the second against each model's own
`bf16`.

### Dense and semi-dense models

Models evaluated on **MegaDepth-1500** and **ScanNet-1500** (1500 pairs each) on **RTX 5070 Ti (`sm_120`)**, PoseLib LO-RANSAC at 1.0 px, 3 seeds `(17, 29, 42)`, `FLASHIM_IMAGE_DECODER=pil`, paired bootstrap 95% CI (`B=10000`) against the `bf16` baseline of the unoptimized code (`fp32` for `orb` and `sift`; `engine="compile"` for `romav2`, `engine="eager"` for all others; working resolution per `benchmarks/accuracy/perf_protocol.py`, e.g. 1200 / 640 px and 4096 kp for classic extractors and LightGlue, 1600 px for `rdd`, 1024 px / 4096 kp for `loma-*`, 800² + 1024² for `romav2`).

| model | MegaDepth base AUC@5 | bf16 Δ | int8 Δ | fp8 Δ | ScanNet base AUC@5 | bf16 Δ | int8 Δ | fp8 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `loftr` | 0.6648 | -0.0005 | -0.0018 | -0.0019† | 0.2181 | +0.0018 | +0.0002 | -0.0018† |
| `romav2` (`compile`) | 0.6231 | -0.0007 | -0.0032† | -0.0044† | 0.3374 | -0.0006 | +0.0010 | -0.0027 |
| `flexglue` (512x512 dense) | 0.6399 | -0.0005 | -0.0012 | -0.0019† | 0.3431 | -0.0002 | -0.0010 | +0.0009 |

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

### Sparse extractors, matchers and pipelines

**84 of the 126 active `(model, preset, dataset)` cells pass, 42 are borderline (`†`: CI straddles -0.005), and 0 fail.** `→ int8` / `→ bf16` mark presets omitted and resolved via `build.NEAREST`; `—` means the model does not define that preset.

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
| `loma-b128` | 0.6756 | +0.0000¹ | +0.0038 | +0.0010 | 0.2980 | +0.0000¹ | -0.0013† | -0.0008† |
| `loma-b` | 0.6746 | +0.0000² | +0.0078 | +0.0019 | 0.3136 | +0.0000² | -0.0011† | -0.0008† |
| `loma-r` | 0.6746 | +0.0000² | +0.0008 | +0.0010 | 0.3023 | +0.0000² | -0.0001† | +0.0008† |
| `loma-l` | 0.6821 | +0.0000² | +0.0011 | -0.0022† | 0.3250 | +0.0000² | -0.0016† | -0.0006† |
| `loma-g` | 0.6789 | +0.0000 | +0.0031 | +0.0008 | 0.3282 | +0.0000 | -0.0018† | -0.0019† |

- ¹ Full MegaDepth-1500 and ScanNet-1500 run: `loma-b128 bf16` is bit-exact with the unoptimized code.
- ² `bf16` is bit-identical to the unoptimized code (outputs and confidences equal on every pair checked: the test pair,
  3 MegaDepth-1500 and 3 ScanNet-1500 pairs; the full-dataset run was not repeated, so the `bf16` cells of these
  variants are `+0.0000` by identity, not by a measurement of their own). `loma-g bf16` is `+0.0000` from the full run.
- **ALIKED family gain**: `aliked`, `raco-aliked`, `aliked-lightglue`, and `raco-aliked-lightglue` score higher than the unoptimized `bf16` because SDDH pixel-space keypoint coordinates now stay in FP32 instead of quantizing to BF16 (2–8 px ULP).

`superpoint-superglue` and `omniglue` share the protocol of the dense table above (1024 px with 1024 keypoints on
MegaDepth; `superpoint-superglue` uses its indoor matcher on ScanNet). `ripepp` and `ripepp-lightglue` use the
sparse-extractor protocol above (1200 px on MegaDepth, 640 x 480 on ScanNet, 4096 keypoints, PoseLib 1.0 px, seeds
17, 29, 42), compared against each model's own `bf16`; `traqpoint` uses 1600 px on MegaDepth and 640 x 480 on ScanNet
with 4096 keypoints (its upstream protocol). `fp8` of `superpoint-superglue`, `omniglue`, and
`ripepp-lightglue` is omitted (-> `int8`): SuperPoint's `conv1b` has 41.6x block-32 channel outliers (-0.013 to
-0.014 AUC@5), OmniGlue's DINOv2 ViT-B/14 loses 0.005-0.006 under MXFP8, and `ripepp-lightglue` drops -0.0538 AUC@5 on
MegaDepth-200 under MXFP8 while running slower than `int8` (66.2 vs 52.6 ms/pair). `ripepp-lightglue` `int8` runs
RIPE++ in `int8` (31/35 convs, `dec.{1,2}` BF16 for subpixel accuracy) with LightGlue in `bf16` (52.6 vs 79.4 ms/pair,
1.51x speedup).

| model | MegaDepth bf16 AUC@5 | int8 Δ | fp8 Δ | ScanNet bf16 AUC@5 | int8 Δ | fp8 Δ |
|---|---:|---:|---:|---:|---:|---:|
| `superpoint-superglue` | 0.5416 | -0.0028† | → int8 | 0.1717 | +0.0068 | → int8 |
| `omniglue` | 0.5195 | +0.0024 | → int8 | 0.1748 | +0.0012† | → int8 |
| `ripepp` | 0.5877 | -0.0040† | +0.0004 | 0.1620 | +0.0020 | -0.0012† |
| `ripepp-lightglue` | 0.6057 | -0.0011† | → int8 | 0.1495 | -0.0009† | → int8 |
| `traqpoint` | 0.6494 | +0.0022 | +0.0028 | 0.2114 | +0.0028 | -0.0020† |

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

The accuracy matrix of the models in the second table of each Accuracy family (about 6.7 hours on an RTX 5070 Ti; resumable, each run acquires a GPU lock
so it can share the GPU) and its report with the paired-bootstrap gate:

```bash
benchmarks/accuracy/b_models_matrix.sh > results/accuracy/b_models/matrix.log 2>&1
uv run python benchmarks/accuracy/b_models_report.py --dir results/accuracy/b_models
```

The accuracy matrix of the models in the first table of each Accuracy family against a base checkout (resumable, supports `OUT_DIR`, `MAX_PAIRS`,
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
