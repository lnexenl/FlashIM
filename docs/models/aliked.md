# ALIKED

Sparse detector + descriptor: a native port of [ALIKED](https://github.com/Shiaoming/ALIKED) (TIM'23). A small conv
encoder (plain 3x3 convs at strides 1/2, deformable 3x3 convs at strides 8/32) feeds a multi-scale feature fusion. A
4-layer conv head produces the keypoint score map, and `SDDH` samples descriptors at deformable offsets around each
detected keypoint (DKD: soft-NMS + soft-argmax sub-pixel refinement).

```python
model = flashim.load("aliked", "bf16", overrides=["variant=aliked-n16"])
feats = model.extract("a.jpg")                       # SparseFeatures: threshold-mode keypoints + 128-d descriptors
m = model.match("a.jpg", "b.jpg")                    # Matches (mutual nearest-neighbour, cosine >= 0.75)
```

For LightGlue matching use `aliked-lightglue` ([LightGlue](lightglue.md)); RaCo keypoints with ALIKED descriptors are
[RaCo + ALIKED](raco-aliked-lightglue.md).

## Variants

Four released checkpoints, chosen with `variant=` and downloaded on first use from the `models/` folder of the ALIKED
repo ([checkpoints](../checkpoints.md)):

| variant | descriptors |
|---|---|
| `aliked-t16` | 64-d |
| `aliked-n16` (default), `aliked-n16rot` | 128-d |
| `aliked-n32` | 128-d, 32 sample points |

## Presets

| preset | what it does |
|---|---|
| `fp32` (default) | the checkpoint's own precision. Dense maps, keypoints and scores match the reference bit-exactly; descriptors to cosine > 0.99999 |
| `bf16` | encoder and `SDDH` in BF16; the score head, NMS / soft-argmax and SDDH pixel-coordinate sampling stay FP32 (preventing 2–8 px BF16 ULP quantization on `kpts_wh`). Deformable convs run their `deform_conv2d` call in FP32 (no BF16 / FP16 kernel exists) |

Every preset folds each block's eval BatchNorm into its convolution. The fold is exact (< 1e-6 max abs diff), and the
folded blocks are listed in `model.fused_batchnorm`. Key hot paths run as custom kernels on CUDA:

- `ops.multiscale_fuse` (Triton, NHWC): fuses the 3 bilinear upsamples, channel concat, FP32 cast and L2 normalize.
- `ops.aliked_score_head` (CUDA C++): fuses the 4-layer score head convs, soft-NMS and soft-argmax.
- `ops.aliked_sample_patches` and `ops.aliked_bilinear_sample` (Triton): SDDH descriptor sampling without `F.unfold` or
  host syncs, keeping pixel coordinates in FP32. Restoring FP32 SDDH pixel coordinates in `bf16` lifted MegaDepth-1500 /
  ScanNet-1500 pose AUC@5 by `+0.0559` / `+0.0068` (`aliked bf16`, RTX 5070 Ti `sm_120`, eager; see
  [benchmarks.md](../benchmarks.md)).

There is no quantized preset (`int8` / `fp8` build `bf16`): the network has no `nn.Linear`, and its small 1x1
projections are memory-bound. The
deformable convs (`block3` / `block4`) can run in INT8 as an option for batched workloads
(`overrides={"optimize.passes+": ["deform_int8"], "optimize.options.deform_int8_calibration": path}`, calibrated with
`DeformConvCalibrator`; feature cosine vs FP32 > 0.999;
`flashim/models/aliked/deform_quant.py`). It is faster than FP32 only from batch 4 on.

`engine="compile"` adds nothing here: the workload is bound by convolutions and `deform_conv2d`, which cuDNN already
schedules well. Eager is the practical default.

## Speed

`extract()` of one 1200 px image, 4096 keypoints, RTX 5070 Ti (`sm_120`), `eager / compile`: `fp32` 5.94 / 5.10 ms, `bf16`
3.72 / 2.70 ms. Method and every other model: [inference benchmark](../benchmarks.md#sparse-extractors).
