# Inference benchmark

Latency of every model, preset and engine, measured on one GPU with one protocol. The tables are latency only: check
[Accuracy](#accuracy) before picking the fastest cell.

- **Setup**: NVIDIA GeForce RTX 5070 Ti (sm_120), torch 2.11.0+cu130, measured 2026-10-06.
- **Timing**: median of 50 calls timed with CUDA events, after 10 untimed warmups (compilation included). One
  configuration per process.
- **Input**: images are decoded and on the GPU before the timing starts. RoMaV2 pairs are preprocessed once; the sparse
  models resize inside the timed call.
- **Units**: milliseconds. The fastest number of a row is in bold.

Presets and engines are described in [models.md](models.md).

## RoMaV2 (dense)

Preloaded forward in both directions, mega1500 setting (800² LR + 1024² HR), B=1. Compiled engines vary 15-25% between
processes, so each is the best of two processes. The speed-up is over `bf16` in the same engine.

| preset | eager | compile | compile-full | speed-up (compile-full) |
|---|---|---|---|---|
| `bf16` | 350.7 | 332.4 | **195.0** | 1.00x |
| `int8` | 140.1 | 133.7 | **104.0** | 1.87x |
| `fp8` | 155.7 | 140.9 | **110.0** | 1.77x |

`int8` and `fp8` were measured under their former names, `int8-fast` and `fp8-fast`. The RoMaV2 table in
[models.md](models.md#romav2) (344 / 110 / 103 ms, speed-up over `bf16` + `compile`) comes from separate runs.

## Sparse extractors

`extract` of one MegaDepth image at 1200 px on the long side, 4096 keypoints. Each cell is `eager / compile`; an empty
cell is a preset the model does not have.

| model | fp32 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| superpoint | 12.7 / 12.5 | 6.63 / 5.57 | 4.85 / **3.60** | 4.73 / 3.79 |
| disk | 14.2 / 11.2 | 8.35 / 7.27 | 9.79 / **7.04** | 7.82 / 7.07 |
| alike | 6.71 / 5.84 | 3.96 / **3.06** | 4.70 / 4.56 | 4.92 / 4.82 |
| xfeat | 2.22 / 2.19 | 2.27 / 1.86 | 2.72 / **1.81** | 2.62 / 2.04 |
| dedode | 197.9 / 125.0 | 56.9 / 61.9 | 46.4 / 50.7 | **41.0** / 51.7 |
| dedode-g | 543.7 / 468.8 | 130.9 / 132.1 | 139.9 / 120.5 | **87.2** / 100.9 |
| dad | 520.2 / 420.1 | 113.2 / 120.9 | 102.4 / 108.5 | **75.1** / 88.7 |
| ripe | 114.4 / 72.5 | 33.6 / fail | **24.5** / fail | 25.1 / fail |
| aliked | 9.05 / 10.3 | 5.34 / **5.17** |  |  |
| raco-aliked | 29.6 / 22.3 | 9.18 / **8.42** |  |  |
| rdd | 69.0 / 61.4 | 34.8 / 25.6 |  | 29.4 / **22.5** |
| rdd `fp8-backbone` | | | | 29.6 / **26.2** |
| rdd `fp8-full` | | | | 28.8 / **22.9** |
| sift | **26.0** / - |  |  |  |
| orb | **10.4** / - |  |  |  |

- RDD's `fp8-backbone` and `fp8-full` presets are in the fp8 column of their own rows.
- SIFT and ORB are OpenCV (fp32, eager only).
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

RoMaV2, RDD and the ALIKED pipelines were not part of that run. Their accuracy is in [models.md](models.md).

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
