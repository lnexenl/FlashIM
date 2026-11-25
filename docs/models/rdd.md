# RDD

Sparse detector + descriptor: a native port of [RDD](https://github.com/xtcpete/rdd) (CVPR'25). A ConvNeXt-tiny backbone
feeds a 5-level multi-scale deformable-attention encoder (4 layers; 140k tokens at 1600 px) that produces the descriptor
map. A small FP32 conv net produces the keypoint score map. `fp32` matches the reference to 1e-5.

```python
model = flashim.load("rdd", "fp8", engine="compile")
feats = model.extract("a.jpg", resize=1600)               # SparseFeatures: 4096 keypoints + 256-d descriptors
m = model.match("a.jpg", "b.jpg", resize=1600)            # Matches (mutual dual-softmax NN)
```

`rdd.pth` and LightGlue's `rdd_lg.pth` are downloaded on first use from the Google Drive folder linked in the RDD README
([checkpoints](../checkpoints.md)). For LightGlue matching use `rdd-lightglue` ([LightGlue](lightglue.md)).

## Presets

| preset | what it does |
|---|---|
| `fp32` | the checkpoint's own precision |
| `bf16` | descriptor branch in BF16; detector, softmax and all sampling / position math in FP32 |
| `fp8` (default) | `bf16` + the encoder's value / output / FFN linears as static MXFP8, run by a fused encoder pass |

There is no `int8` preset (it builds `bf16`). MXFP8 sampling-offset / attention-weight projections (AUC@5 0.5242) and
ConvNeXt MLPs were measured no faster than `fp8` (the sampling-offset projections share one BF16 GEMM there; the
ConvNeXt MLP GEMMs are small-K and memory-bound) and were dropped.

The fused encoder pass of `fp8` combines deformable attention with the softmax fused in (one BF16 GEMM for sampling
offsets + attention logits), biases in the GEMM epilogue, LayerNorm→quantize producers, and one MXFP8 MLP kernel for
`linear1` + bias + GELU + quantize + `linear2` (the 1024-wide hidden activation never leaves the SM; bit-identical to the
two-GEMM path).

The ConvNeXt-tiny backbone runs in `channels_last` with folded `Block.gamma` and zero-copy NHWC LayerNorm, the detector
and matchibility branches fold eval BatchNorm (`rdd.detector` wall time `8.56 -> 5.78 ms` at 1600 px on RTX 5070 Ti
`sm_120`, `eager`), and `match_features` uses Triton `ops.dual_softmax_mutual` (`4096x4096`, `bf16`, RTX 5070 Ti
`sm_120`, `eager`: `5.76 -> 0.75 ms`, 7.67x).

`engine="compile"` compiles the backbone, input projections, detector and encoder once per input resolution.
`precision.overrides.detector=bf16` runs the detector's conv trunk in BF16 (score head FP32); it is off by default because it
changes about 12% of the 4096 keypoints.

## Speed

`extract()` at 1056x1600, image preloaded, keypoints + descriptors included, RTX 5070 Ti (`sm_120`, 2026-10 baseline):

| configuration | extract | peak memory |
|---|---|---|
| RDD reference (PyTorch deformable path, no CUDA extension) | OOM (16 GB) | |
| `fp32` eager | 110.9 ms | 2.0 GiB |
| `bf16` eager | 57.6 ms | 1.1 GiB |
| `bf16` + `compile` | 40.3 ms | 1.1 GiB |
| `fp8` eager | 49.3 ms | 1.3 GiB |
| `fp8` + `compile` | 35.4 ms | 1.1 GiB |

In the 2026-11 `sm_120` eager sweep (1600 px `toronto_A`, RTX 5070 Ti `sm_120`, `eager`), ConvNeXt `channels_last` +
NHWC LayerNorm cut `extract()` by `5.8 ms` (`bf16` `62.72 -> 56.92 ms` [1.10x], `fp8` `53.59 -> 47.76 ms` [1.12x]);
the `compile` rows above were not re-measured.

## Accuracy

MegaDepth-1500 pose AUC, RDD protocol (1600 px, 4096 keypoints, mutual NN, OpenCV RANSAC, RTX 5070 Ti `sm_120`, `eager`
unless noted), RGB input:

| configuration | AUC@5 | AUC@10 | AUC@20 |
|---|---|---|---|
| `fp32` | 0.5216 | 0.6827 | 0.8019 |
| `bf16` | 0.5270 | 0.6855 | 0.8031 |
| `fp8` | 0.5418 | 0.6974 | 0.8137 |
| `fp8` + `compile` | 0.5287 | 0.6887 | 0.8071 |

The run-to-run spread of these sparse pipelines is about ±0.01, so FP8 costs no measurable accuracy. Under the 2026-11
MegaDepth-1500 / ScanNet-1500 paired-bootstrap gate (RTX 5070 Ti `sm_120`, `eager`, 3 seeds, PoseLib 1.0 px), both
`rdd bf16` and `rdd fp8` pass on both datasets ([benchmarks.md](../benchmarks.md#accuracy-of-the-pre-2026-11-models-2026-11-gate-sm_120)).
RDD's own script feeds OpenCV BGR images; `fp32` scores 0.5155 AUC@5 that way. Run with `benchmarks/accuracy/run_sparse.py`;
the matrix is `benchmarks/accuracy/rdd_matrix.sh`.
