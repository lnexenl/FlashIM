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

## Accuracy

A faster cell is not always a usable one. This is the pose AUC@5 change of each quantized preset against the same
model's `bf16` on MegaDepth-1500 (4 RANSAC seeds, eager, run 2026-10-05). The gate is a drop below 0.005; cells that
fail it are in bold.

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

RoMaV2, RDD and the ALIKED pipelines were not part of that run. Their accuracy is on their model pages ([models.md](models.md)).

## Accuracy of the 2026-11 additions

Pose AUC@5 of `roma`, `tiny-roma`, `minima-roma`, `mast3r`, `dust3r`, `ufm`, `eloftr`, `xoftr`, `minima-loftr`,
`superpoint-superglue` and `omniglue`, on MegaDepth-1500 and ScanNet-1500 (1500 pairs each, eager). Each cell is the
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
uv run python benchmarks/speed/extractor.py --model disk-lightglue --preset fp8 --resize 1200 \
    --overrides components.extractor.preset=fp8
```

The accuracy matrix of the 2026-11 additions (about 6.7 hours; resumable, one `flock` on `~/.cache/flashim/gpu.lock`
per run) and its report with the paired-bootstrap gate:

```bash
setsid nohup benchmarks/accuracy/b_models_matrix.sh > results/accuracy/b_models/matrix.log 2>&1 &
uv run python benchmarks/accuracy/b_models_report.py --dir results/accuracy/b_models
```

`benchmarks/accuracy/run_sparse.py --dataset scannet1500` is the generic ScanNet-1500 runner (any model that returns
`Matches`; `run.py` is RoMaV2's).
