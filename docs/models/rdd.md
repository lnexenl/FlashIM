# RDD

Sparse detector + descriptor: a native port of [RDD](https://github.com/xtcpete/rdd) (CVPR'25). A ConvNeXt-tiny backbone
feeds a 5-level multi-scale deformable-attention encoder (4 layers; 140k tokens at 1600 px) that produces the descriptor
map. A small FP32 conv net produces the keypoint score map. `fp32` matches the reference to 1e-5.

```python
model = flashim.load("rdd", preset="fp8", engine="compile")
feats = model.extract("a.jpg", resize=1600)               # SparseFeatures: 4096 keypoints + 256-d descriptors
m = model.match("a.jpg", "b.jpg", resize=1600)            # SparseMatchResult (mutual dual-softmax NN)
```

`rdd.pth` and LightGlue's `rdd_lg.pth` are downloaded on first use from the Google Drive folder linked in the RDD README
([checkpoints](../checkpoints.md)). For LightGlue matching use `rdd-lightglue` ([LightGlue](lightglue.md)).

## Presets

| preset | what it does |
|---|---|
| `fp32` | the checkpoint's own precision |
| `bf16` | descriptor branch in BF16; detector, softmax and all sampling / position math in FP32 |
| `fp8` (default) | `bf16` + the encoder's value / output / FFN linears as static MXFP8, run by a fused encoder pass |
| `fp8-full` | `fp8` + MXFP8 sampling-offset / attention-weight projections |
| `fp8-backbone` | `fp8` + MXFP8 ConvNeXt MLPs |

`fp8-full` and `fp8-backbone` are no faster than `fp8` (the sampling-offset projections share one BF16 GEMM there; the
ConvNeXt MLP GEMMs are small-K and memory-bound) and are kept for experiments.

The fused encoder pass of `fp8` combines deformable attention with the softmax fused in (one BF16 GEMM for sampling
offsets + attention logits), biases in the GEMM epilogue, LayerNorm→quantize producers, and one MXFP8 MLP kernel for
`linear1` + bias + GELU + quantize + `linear2` (the 1024-wide hidden activation never leaves the SM; bit-identical to the
two-GEMM path).

`engine="compile"` compiles the backbone, input projections, detector and encoder once per input resolution.
`detector_dtype="bfloat16"` runs the detector's conv trunk in BF16 (score head FP32); it is off by default because it
changes about 12% of the 4096 keypoints.

## Speed

`extract()` at 1056x1600, image preloaded, keypoints + descriptors included, RTX 5070 Ti:

| configuration | extract | peak memory |
|---|---|---|
| RDD reference (PyTorch deformable path, no CUDA extension) | OOM (16 GB) | |
| `fp32` eager | 110.9 ms | 2.0 GiB |
| `bf16` eager | 57.6 ms | 1.1 GiB |
| `bf16` + `compile` | 40.3 ms | 1.1 GiB |
| `fp8` eager | 49.3 ms | 1.3 GiB |
| `fp8` + `compile` | 35.4 ms | 1.1 GiB |

## Accuracy

MegaDepth-1500 pose AUC, RDD protocol (1600 px, 4096 keypoints, mutual NN, OpenCV RANSAC), RGB input:

| configuration | AUC@5 | AUC@10 | AUC@20 |
|---|---|---|---|
| `fp32` | 0.5216 | 0.6827 | 0.8019 |
| `bf16` | 0.5270 | 0.6855 | 0.8031 |
| `fp8` | 0.5418 | 0.6974 | 0.8137 |
| `fp8` + `compile` | 0.5287 | 0.6887 | 0.8071 |
| `fp8-full` | 0.5242 | 0.6801 | 0.7980 |

The run-to-run spread of these sparse pipelines is about ±0.01, so FP8 costs no measurable accuracy. RDD's own script
feeds OpenCV BGR images; `fp32` scores 0.5155 AUC@5 that way. Run with `benchmarks/accuracy/run_sparse.py`; the matrix is
`benchmarks/accuracy/rdd_matrix.sh`.
