# RoMaV2

Dense matcher: a native port of RoMa v2 (DINOv3 ViT-L/16, matcher ViT, DPT head, conv refiners).

```python
model = flashim.load("romav2", preset="fp8", setting="mega1500", engine="compile-full")
result = model.match("a.jpg", "b.jpg")                       # DenseMatchResult (warp, overlap, precision)
matches, conf, prec_a, prec_b = model.sample(result, 5000)
kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
```

## Presets

| preset | what it does |
|---|---|
| `bf16` | the reference, bit-identical to the RoMaV2 implementation |
| `int8` | RoMaV2's `int8_*` recipe: symmetric per-output-channel weights, static per-tensor activation scales (99.99th-percentile observer, 64 calibration pairs), INT32 accumulation with a fused dequantize + bias epilogue. Covers the transformer linears, the DPT head's four 1x1 input projections and the refiner blocks at strides 4 and 2 |
| `fp8` (default) | MXFP8. Covers every DINOv3 and matcher-ViT linear (static calibrated activation scales, fused LayerNorm→quantize and GELU→quantize producers) and the refiner blocks at strides 4 and 2 (one Triton kernel for depthwise 5x5 + bias + ReLU + dynamic quantization, then a block-scaled MXFP8 pointwise GEMM) |

`int8` and `fp8` share FlashIM's refiner kernels: fused BF16 stride-1 conv blocks, a fused refiner input (warp sampling,
displacement embedding and local correlation in one pass), fused FP32 output heads, and both match directions refined as
one batch. Neither is bit-identical to RoMaV2 (reduction order, rounding points); both are validated by pose AUC.

Coordinate heads, the patch embedding, refiner projections and displacement embeddings are protected: a quantization
config that targets them is rejected.

## Speed

mega1500 setting (800² LR + 1024² HR), both directions, B=1, preloaded forward, RTX 5070 Ti:

| preset | forward | vs `bf16` |
|---|---|---|
| `bf16` | 195.0 ms | 1.00x |
| `int8` | 104.0 ms | 1.87x |
| `fp8` | 110.0 ms | 1.77x |

Method and the full table: [inference benchmark](../benchmarks.md#romav2-dense).

## Accuracy

Pose AUC@5 change vs `bf16` (3 seeds). The gate is a 3-seed mean of at least -0.005; RANSAC seed noise is about 0.003.

| preset | MegaDepth-1500 | ScanNet-1500 |
|---|---|---|
| `fp8` | -0.0023 | +0.0007 |
| `int8` | -0.0010 | +0.0004 |

HardMatch (LoMa, ECCV 2026: 900 hand-labelled hard pairs, mAA@10px, 3 seeds, `wxbs` setting; run with
`benchmarks/accuracy/run_hardmatch.py`):

| | mAA@10px | vs `bf16` |
|---|---|---|
| `bf16` | 0.4786 | |
| RoMaV2's own FP8 recipe | 0.4707 | -0.0079 |
| `fp8` | 0.4712 | -0.0074 |

FP8 costs more on these deliberately hard pairs than on MegaDepth / ScanNet; FlashIM's extra kernels add none of that
cost. HardMatch was not run for `int8`.

## Options

- **Match sampling.** `sample()` (KDE-balanced) takes 0.86 ms, against 16 ms and 1.5 GiB of scratch memory in RoMaV2,
  with the same pose AUC within noise. `FLASHIM_FAST_SAMPLING=0` restores RoMaV2's. `match(..., num_samples=N)` returns
  pixel keypoints directly.
- **INT8 attention.** `flashim.load("romav2", preset="int8", attention="sage-fp8")` runs the attention of every
  quantized transformer block through [SageAttention](https://github.com/thu-ml/SageAttention) (INT8 Q/K with K
  smoothing, FP8 PV; `"sage"`: FP16 PV). It needs `sageattention` 2.x built for your GPU, which is not a project
  dependency (with GCC 16, build it with clang).

  | `int8` attention | forward | AUC@5 vs `bf16`, MegaDepth / ScanNet |
  |---|---|---|
  | `sdpa` (default) | 102.7 ms | |
  | `sage` | 99.7 ms | -0.0012 / -0.0002 |
  | `sage-fp8` | 95.4 ms | +0.0008 / +0.0010 |

- **Calibration.** `scripts/calibrate_romav2.py --format int8 --preset int8 [--merge]`. The shipped table is
  `flashim/models/romav2/calibration/romav2_int8_mega1500.safetensors`.
