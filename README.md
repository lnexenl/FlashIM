# FlashIM

Fast, quantized, multi-platform inference for image-matching models.

```python
import flashim

model = flashim.load("romav2", preset="fp8", setting="mega1500", engine="compile")
result = model.match("a.jpg", "b.jpg")                 # DenseMatchResult (warp, overlap, precision)
matches, conf, prec_a, prec_b = model.sample(result, 5000)
kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
```

## Configuring a run

Every model is built by one generic builder from a `RunConfig` (`flashim/config.py`): `arch`, `precision`
(dtype, quantization, calibration), `optimize` (named passes, attention impl, strict), `engine`, `inference`
(per-call defaults) and, for pipelines, `components`. A preset is a named partial config; files, dicts and dotted
overrides layer on top (later wins). The keyword API above is a thin layer over the same builder.

```python
model = flashim.load("romav2", "fp8", overrides=["engine.kind=compile-full", "arch.setting=mega1500"])
model = flashim.load("raco-aliked-lightglue", "int8", config="run.toml")   # .toml / .json (.yaml with PyYAML)
flashim.resolve("rdd-lightglue", "fp8").to_dict()      # the config a build would use, without building
model.run_config.digest(), model.build_report          # what was asked for / what was actually applied
```

```toml
# run.toml: RaCo+ALIKED in BF16, LightGlue INT8, everything compiled
device = "cuda"
[engine]
kind = "compile"
[components.extractor]
preset = "bf16"
[inference]
resize = 1024
top_k = 1024
```

Overrides: `optimize.passes+=x` appends a pass, `components.matcher.optimize.attention=sage` reaches into a
pipeline component. Unknown keys, passes, quant configs and engines are errors; a pass the platform cannot run is
skipped with a warning (`optimize.strict=true` raises).

## Models

| model | presets | notes |
|---|---|---|
| `romav2` | `bf16`, `int8`, `fp8` (default) | native port of RoMa v2; `bf16` is bit-identical to the RoMaV2 reference implementation (so are the `fp8-p8` / `fp8-s4` / `fp8-s42` oracle presets the parity tests use) |
| `rdd` | `fp32`, `bf16`, `fp8` (default), `fp8-full`, `fp8-backbone` | native port of [RDD](https://github.com/xtcpete/rdd) (CVPR'25, sparse detector + descriptor); `fp32` matches the reference to 1e-5 (see [RDD](#rdd)) |
| `aliked` | `fp32` (default), `bf16` | native port of [ALIKED](https://github.com/Shiaoming/ALIKED) (TIM'23, sparse detector + descriptor); `fp32` matches the reference bit-exactly (see [ALIKED](#aliked)) |
| `superpoint`, `alike`, `disk`, `xfeat`, `ripe`, `dedode`, `dedode-g`, `dad`, `loftr` | `fp32`, `bf16` (default), `int8`, `fp8` | native ports, each verified against its upstream implementation (see [More sparse and semi-dense models](#more-sparse-and-semi-dense-models)) |
| `superpoint-lightglue`, `disk-lightglue`, `xfeat-lightglue` | `fp32`, `bf16`, `int8`, `fp8` | extractor + LightGlue (cvg weights; XFeat's LighterGlue), matcher preset selects the pair, `components.extractor.preset` the extractor |
| `sift`, `orb` | `fp32` | OpenCV baselines (no weights: nothing to quantize) |

RoMaV2 presets:

- `bf16` — the reference.
- `fp8` — MXFP8 (E4M3 values, E8M0 scale per 32 elements): every DINOv3 / matcher-ViT linear with *static*
  calibrated activation scales (fused LayerNorm→quant and GELU→quant producers), fused stride-4 and stride-2
  refiner blocks (one Triton kernel for depthwise 5x5 + bias + ReLU + *dynamic* MXFP8 quantization, then a
  block-scaled MXFP8 pointwise GEMM), and FlashIM refiner kernels: fused BF16 stride-1 conv blocks, a fused refiner
  input (warp sampling, displacement embedding and local correlation in one pass), fused FP32 output heads and both
  match directions refined as one batch. Not bit-identical to RoMaV2 (reduction order / rounding points); validated
  by pose AUC.
- `int8` — RoMaV2's `int8_*` recipe (symmetric per-output-channel int8 weights, *static* per-tensor activation
  scales calibrated with the 99.99th-percentile observer over 64 pairs, INT32 accumulation with a fused
  dequantize + bias epilogue; Triton GEMM, `torch._int_mm` kept as an alternative implementation) on the
  transformer linears and the DPT head's four 1x1 input projections, INT8 refiner blocks at strides 4/2 (stride 1
  stays fused BF16; C=128 blocks run as one fused depthwise + INT8 GEMM kernel, C=512 producers as TileLang) and the
  same FlashIM refiner kernels as `fp8`. Not RoMaV2-exact.

Compile is an engine, not a preset: any preset runs with `engine="eager"`, `"compile"` or `"compile-full"`. The oracle
presets `fp8-p8` (transformer linears only, BF16 attention and refiners), `fp8-s4` and `fp8-s42` (+ fused stride-4 /
stride-4 and 2 refiner blocks) reproduce RoMaV2's FastFP8 runtime bit for bit and exist for the parity tests.

Optional INT8 attention: `flashim.load("romav2", preset="int8", attention="sage-fp8")` (or `"sage"`: INT8 QK +
FP16 PV) runs every quantized transformer block's attention through [SageAttention](https://github.com/thu-ml/SageAttention)
(INT8 Q/K with K smoothing, FP8/FP16 PV). Needs `sageattention` 2.x built for your GPU (not a project dependency;
with GCC 16 build it with clang). `int8` + `compile-full`: 102.7 ms -> 99.7 (`sage`) / 95.4 ms (`sage-fp8`);
AUC@5 vs `bf16` (3 seeds, Mega / ScanNet): `sage` -0.0012 / -0.0002, `sage-fp8` +0.0008 / +0.0010 (noise ~0.003).

Calibrate with `scripts/calibrate_romav2.py --format int8 --preset int8 [--merge]`; the shipped table is
`flashim/models/romav2/calibration/romav2_int8_mega1500.safetensors`.

Engines: `eager`; `compile` (DINOv3 + matcher ViT, RoMaV2's validated setting); `compile-full`
(also VGG with folded BatchNorm, the DPT head and the refiner conv stacks).

Measured on an RTX 5070 Ti (mega1500 setting, 800² LR + 1024² HR, bidirectional, B=1, preloaded forward):

| configuration | forward | vs BF16 |
|---|---|---|
| `bf16` + `compile` | 344 ms | 1.0x |
| RoMaV2 FastFP8 (s4+s2 fused) + compile | 210 ms | 1.64x |
| `fp8` + `compile-full` | 110 ms | 3.13x |
| `int8` + `compile-full` | 103 ms | 3.36x |

(`int8` is faster than `fp8`.)

Pose AUC@5 change vs `bf16` (3 seeds, `compile-full`, Mega1500 / ScanNet1500): `fp8` -0.0023 / +0.0007 (gate:
3-seed mean >= -0.005; every `fp8` seed is within -0.005); `int8` -0.0010 / +0.0004 (inside the RANSAC seed noise
of ~0.003). HardMatch not run for INT8.

HardMatch (LoMa, ECCV 2026; 900 hand-labelled hard pairs, mAA@10px, 3 seeds, `wxbs` setting):
`bf16` 0.4786, `fp8-s42` (RoMaV2's own FP8 recipe) 0.4707 (-0.0079), `fp8` 0.4712 (-0.0074). FP8's cost is larger on these
deliberately hard pairs than on MegaDepth/ScanNet; FlashIM's extra kernels add none of it.
Run with `benchmarks/accuracy/run_hardmatch.py` (official `hardmatch` package protocol).

`sample()` (KDE-balanced match sampling) takes 0.86 ms (RoMaV2: 16 ms, 1.5 GiB scratch); pose AUC
vs RoMaV2's sampling +0.0012 / -0.0010 (3 seeds; `FLASHIM_FAST_SAMPLING=0` restores it); `match(..., num_samples=N)` returns pixel keypoints.

Coordinate heads, the patch embedding, refiner projections and displacement
embeddings are protected: a quantization config that targets them is rejected.

## RDD

```python
model = flashim.load("rdd", preset="fp8", engine="compile")  # checkpoint: see flashim/models/rdd/weights.py
feats = model.extract("a.jpg", resize=1600)               # SparseFeatures: 4096 keypoints + 256-d descriptors
m = model.match("a.jpg", "b.jpg", resize=1600)            # SparseMatchResult (mutual dual-softmax NN)
```

ConvNeXt-tiny backbone -> 5-level multi-scale deformable-attention encoder (4 layers; 140k tokens at 1600 px) ->
descriptor map; a small FP32 conv net produces the keypoint score map. `rdd.pth` (and LightGlue's `rdd_lg.pth`)
is downloaded on first use from the Google Drive folder linked in the RDD README (see [Checkpoints](#checkpoints)).

- `bf16` — descriptor branch in BF16; detector, softmax and all sampling / position math in FP32.
- `fp8` — `bf16` + the encoder's value / output / FFN linears as static MXFP8, run by a fused encoder pass:
  deformable attention with the softmax fused in (one BF16 GEMM for sampling offsets + attention logits), GEMM-epilogue
  biases, LayerNorm->quant producers, and one fused MXFP8 MLP kernel (`linear1` + bias + GELU + quantize + `linear2`,
  the 1024-wide hidden activation never leaves the SM; bit-identical to the two-GEMM path).
- `fp8-full` — `fp8` + MXFP8 sampling-offset / attention-weight projections (no faster: they share one BF16 GEMM in `fp8`).
- `fp8-backbone` — `fp8` + MXFP8 ConvNeXt MLPs. Slower: those GEMMs are small-K and memory-bound (kept for experiments).

`engine="compile"` compiles backbone, input projections, detector and encoder once per input resolution.

The keypoint detector stays FP32 by default. `detector_dtype="bfloat16"` runs its conv trunk in BF16 (score head FP32):
−0.9 ms compiled (3.7 -> 2.7 ms), but ~12% of the 4096 keypoints change and MegaDepth AUC@5/@10/@20 moved
−0.012/−0.015/−0.014 (eager) and −0.004/−0.007/−0.006 (compile) vs the FP32 detector — inside run noise, but every
paired number went down.

`extract()` at 1056x1600 on an RTX 5070 Ti (image preloaded, keypoints + descriptors included):

| configuration | extract | peak memory |
|---|---|---|
| RDD reference (PyTorch deformable path, no CUDA extension) | OOM (16 GB) | — |
| `fp32` eager | 110.9 ms | 2.0 GiB |
| `bf16` eager | 57.6 ms | 1.1 GiB |
| `bf16` + `compile` | 40.3 ms | 1.1 GiB |
| `fp8` eager | 49.3 ms | 1.3 GiB |
| `fp8` + `compile` | 35.4 ms | 1.1 GiB |

MegaDepth-1500 pose AUC@5 / @10 / @20 (RDD protocol: 1600 px, 4096 keypoints, mutual NN, OpenCV RANSAC; RGB input —
RDD's own script feeds OpenCV BGR, `fp32` 0.5155 that way): `fp32` 0.5216 / 0.6827 / 0.8019, `bf16` 0.5270 / 0.6855 /
0.8031, `fp8` 0.5418 / 0.6974 / 0.8137, `fp8` + `compile` 0.5287 / 0.6887 / 0.8071, `fp8-full` 0.5242 / 0.6801 / 0.7980.
Run-to-run spread of these sparse pipelines is about ±0.01 (`bf16` eager vs `compile` differ by 0.006): FP8 costs no
measurable accuracy. `benchmarks/accuracy/run_sparse.py`, matrix `benchmarks/accuracy/rdd_matrix.sh`.

## ALIKED

```python
model = flashim.load("aliked", preset="fp32", variant="aliked-n16")  # checkpoint: see flashim/models/aliked/weights.py
feats = model.extract("a.jpg")                               # SparseFeatures: threshold-mode keypoints + 128-d descriptors
m = model.match("a.jpg", "b.jpg")                            # SparseMatchResult (mutual nearest-neighbour, cosine >= 0.75)
```

A small conv encoder (plain 3x3 convs at strides 1/2, deformable 3x3 convs at strides 8/32) feeds a multi-scale
feature fusion; a 4-layer conv head produces the keypoint score map and `SDDH` samples descriptors at
deformable offsets around each detected keypoint (DKD: soft-NMS + soft-argmax sub-pixel refinement). Four
released checkpoints (`variant=`): `aliked-t16` (64-d), `aliked-n16` / `aliked-n16rot` (128-d, default),
`aliked-n32` (128-d, 32 sample points), downloaded on first use from the `models/` folder of the ALIKED repo
(see [Checkpoints](#checkpoints)).

- `fp32` — the checkpoint's own precision; matches the reference implementation's dense maps, keypoints and
  scores bit-exactly, descriptors to cosine > 0.99999 (deformable-sampling summation order only).
- `bf16` — encoder and `SDDH` in BF16 (deformable convs run their `deform_conv2d` call in FP32: no BF16/FP16
  kernel exists); the score head and NMS / soft-argmax / sampling math stay FP32.

Every preset folds each block's eval BatchNorm into its convolution (`fuse.fold_batchnorm`; exact, not an
approximation — verified <1e-6 max abs diff) and reports the folded blocks as `model.fused_batchnorm`.
`load_aliked(..., channels_last=True)` is available but off by default: measured *slower* on an RTX 5070 Ti
(ALIKED's channel counts are tiny, <=128, and `deform_conv2d` always returns contiguous NCHW regardless of
its input's layout, so cuDNN has no channels-last algorithm to pick for most of the encoder).
`extract()` at 1024x1024 on an RTX 5070 Ti (image preloaded, `aliked-n16`, keypoints + descriptors included):

| configuration | extract |
|---|---|
| `fp32` eager, no BatchNorm fold | 15.3 ms |
| `bf16` eager, no BatchNorm fold | 15.1 ms |
| `bf16` eager, BatchNorm folded (default) | 10.5 ms |
| `bf16` + `compile`, BatchNorm folded | 10.3 ms |

`engine="compile"` gives no further win here beyond BF16 + the BatchNorm fold (this workload is
conv/`deform_conv2d`-bound, which cuDNN already schedules well; unlike RoMaV2's transformer-heavy graph,
there is little for Inductor's fusion to add) — kept available, but eager is the practical default.

No quantized preset by default. ALIKED has no `nn.Linear` layers; the small 1x1 projections
(`conv1..4`, the `downsample`s, `desc_head.sf_conv`, all K/N <= 128) are the only convs the existing
quantization infra can otherwise target, and RDD's `fp8-backbone` preset already found that class of GEMM
(small-K, memory-bound) *slower* to quantize than to leave in BF16 — ALIKED's are smaller still, so it
wasn't re-measured.

The deformable convs (`block3`/`block4`, where most of the runtime actually goes) *were* quantized: a new
op, `ops.deform_patches_int8`, gathers each output pixel's receptive-field patch at its (bilinearly
interpolated, data-dependent) offset and quantizes it to static INT8 in one fused Triton kernel — feeding
the existing INT8 GEMM (`flashim/models/aliked/deform_quant.py`). Correctness is good (feature cosine
> 0.999 vs FP32; the sampling offsets themselves stay FP32 — see the module's docstring for why) and it's
fully calibratable (`DeformConvCalibrator`, `load_aliked(..., deform_int8_calibration=...)`).

First cut was **10-35% slower** than the FP32 `torchvision.ops.deform_conv2d` call it replaces, at every
batch size tried. Profiling (not assumption) showed why: the INT8 GEMM itself was cheap (~7us); the
*gather* kernel was 4x slower than torchvision's, because `x` is NCHW, so gathering one pixel's channels
strides across `H*W`-sized planes with no memory coalescing at all. Converting `x` to channels-last inside
the Triton kernel fixed most of it: on an RTX 5070 Ti, `batch=1` is now within noise of FP32 (~2%), and
`batch>=4` is a clear **win** (25-39% faster, growing with batch). Kept opt-in — worth it for batched
workloads, not for ALIKED's typical single-image `extract()`.

`ops.multiscale_fuse` fuses `dense()`'s multi-scale feature step — 3 bilinear upsamples + a channel concat +
the FP32 cast (`score_head`'s input) + the L2 normalize (the descriptor feature map) — into one Triton
kernel and one pass (safe because `dim // 4 <= 32` for every released variant, so a kernel program already
holds all `4 * dim // 4` output channels of its pixel tile in registers). First cut used FP64 for the
bilinear coordinate math (to match `F.interpolate`'s own precision) and pre-cast every input to FP32 before
the kernel — both measured mistakes: FP64 has very limited throughput on consumer GPUs, and the pre-cast
redid a full memory pass over the biggest input. Fixed (FP32 coordinates, in-kernel upcast-on-load): `bf16`
`extract()` at 1024x1024 (`aliked-n16`) went from 6.97 ms to 5.94 ms; the full ALIKED+LightGlue pipeline
below from 16.6 ms to 14.6 ms. Numerics unaffected (parity tests still bit-exact FP32 / cosine > 0.9999 BF16).

## ALIKED + LightGlue

```python
pipeline = flashim.load("aliked-lightglue", preset="fp32", aliked_preset="fp32")  # LightGlue weights: cvg/LightGlue's aliked_lightglue.pth
m = pipeline.match("a.jpg", "b.jpg", resize=1024, top_k=1024)
```

cvg/LightGlue's weights trained on ALIKED's own keypoints and descriptors (`aliked_lightglue.pth`;
`input_dim=128 -> dim=256` via `input_proj`, `filter_threshold=0.1`, see `lightglue.config.ALIKED_CONFIG`);
`variant="aliked"` in `load_lightglue()`, or the shortcut `load_aliked_lightglue()` / `flashim.load("aliked-lightglue")`.
cvg/LightGlue trains one LightGlue per (detector, descriptor) pair, so the RaCo pairing below has its own weights
(`variant="raco-aliked"`). The `int8`/`fp8` presets use calibration tables computed on each pairing's features
(`lightglue/calibration/lightglue_{aliked,raco-aliked}_{int8,fp8}_mega.safetensors`). MegaDepth-1500 AUC@5 (unified
protocol, 4 seeds, fp32 ALIKED): fp32 0.627, bf16 0.627, int8 0.620, fp8 0.613 (RaCo + ALIKED: bf16 0.640, int8 0.638,
fp8 0.634); before the weights were corrected this pipeline scored 0.595 in fp32.

## RaCo + ALIKED + LightGlue

```python
pipeline = flashim.load("raco-aliked-lightglue", preset="bf16", extractor_preset="bf16", engine="compile")
m = pipeline.match(a, b, top_k=1024, threshold=0.1)   # images must be 32-aligned, or pass resize=
```

The `raco_aliked` pipeline of [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX), ported natively
(`flashim/models/raco`). It is **not** the plain ALIKED pipeline above: [RaCo](https://github.com/cvg/RaCo) is a separate
detector network (ALIKED-style CNN -> NMS -> 2K candidates with soft-argmax) plus a **ranker** (9 residual blocks at full
resolution) that reorders the candidates and keeps the top K; ALIKED then only *describes* those keypoints (encoder +
SDDH, no score head/DKD) and LightGlue matches them, with cvg/LightGlue's weights for this pairing
(`raco_aliked_lightglue.pth`, trained on RaCo's keypoints). Keypoints and ranker scores are bit-exact against their PyTorch
modules in FP32 (identical keypoint set, `tests/parity/raco`), and the full pipeline returns the same 172 matches.

Speed, same 1024x1024 pair, 1024 keypoints, threshold 0.1, RTX 5070 Ti, decode/resize excluded:

| pipeline | ranker | time/pair |
|---|---|---|
| LightGlue-ONNX, TensorRT 11.3 fp16 | dense (exact) | 18.1 ms |
| LightGlue-ONNX, TensorRT 11.3 fp16 | `boundary` (their default: reranks only a 512-candidate window) | 13.8 ms |
| FlashIM, bf16 + `compile` | dense (exact) | **14.6 ms** |
| FlashIM, bf16 eager | dense (exact) | 19.9 ms |
| FlashIM, fp32 eager (TF32 convs) | dense (exact) | 57.3 ms |

On the same (exact) algorithm FlashIM is now ~20% faster than their TensorRT-fp16 engine (their number is engine-only GPU
time; FlashIM's includes Python launch overhead). FlashIM's `boundary` ranker is not implemented, so the 13.8 ms number
still has no like-for-like counterpart. The port started at 88.8 ms; what moved it, in order of size:

- **Fused multi-scale kernel** (`ops.multiscale_fuse`): upsample x3 + concat + the score head's first layer (1x1 conv 128->8
  + SELU, FP32-accurate `tf32x3`) + L2 normalize in one pass, never materializing the 512 MB FP32 concat. The first cut was
  2.3 ms; Nsight Compute showed L1/TEX at 93% (DRAM 5%) because the bilinear gathers read NCHW maps, so each warp
  touched 32 cache lines. Making the coarse maps channels-last: **2.3 -> 0.41 ms** (0.57 ms for ALIKED's descriptor path).
- **BatchNorm folding** (exact) and a **bf16 detector trunk + ranker** under `compile`.
- **Fused ranker blocks**: each residual block is one kernel (input read once, output written once, intermediate kept on
  chip) plus a fused 5x5 reflect head. Three implementations of the same block, all agreeing to ~1e-5 relative, per
  12-channel block at 1024x1024: unfused torch/Inductor ~0.33 ms, Triton 0.237 ms (`backends/triton/raco.py`; per-tile
  L1/L2 scratch + barriers), **CUDA C++ 0.175 ms** (`backends/cuda/csrc/raco_resblock.cu`, used by default). The CUDA kernel is
  an implicit GEMM with `mma.sync.m16n8k16`: A-fragments are `ldmatrix`-ed straight from the NHWC input/intermediate
  tile in shared memory (no im2col), 28 KB of shared memory so several blocks fit per SM, shortcut as a tenth tap. It is
  JIT-built with `nvcc` on first use (cached in `~/.cache/flashim/cuda`, pure CUDA with no torch headers) and falls back
  to the Triton kernel when there is no toolchain (`RaCo.use_fused_ranker(backend="auto"|"cuda"|"triton")`). Ranker:
  3.0 -> 2.3 -> ~1.7 ms per image, and closer to FP32 than the compiled bf16 path (99.8% of keypoints unchanged vs 99.6%).
  A TileLang version (correct, but 0.43 ms: one 4-warp block per SM from a 51 KB im2col buffer) was tried and dropped.
  Padded channels/halo make ~2x the MACs wasted in every version; that, not the language, is the remaining ceiling.

Accuracy of the whole fast path against the exact FP32 reference on the test pair: 98.5-98.9% of the keypoints are the
same pixels, RANSAC inliers are within run-to-run noise (73-81 across FP32/TF32/bf16 variants), and bf16 descriptors give
~9% fewer matches (155 vs 170) at the same inlier count.

Overlapping those CNNs on CUDA streams (`RaCoALIKED.concurrent`, off by default) gave no speedup: the GPU is already
saturated at 1024x1024. It also exposed that **`torchvision.ops.deform_conv2d` ignores the current CUDA stream** and
returns wrong values when its input was produced on a side stream (ALIKED's `block3`/`block4`), so ALIKED's encoder is
kept on the caller's stream.

INT8/FP8: LightGlue's quantized presets do not help at 1024 keypoints (bf16 13.8 ms vs int8 14.6 ms vs fp8 15.3 ms in
the plain-ALIKED pipeline; measured before each pairing had its own calibration table). LightGlue-ONNX's own ModelOpt FP8
engines are not comparable: they also quantize RaCo (56-59 matches vs ~165 in FP16) and were calibrated on 3 pairs.
Their INT8 quantization was not completed.

Getting their engines running needed `onnxslim` (constant-folds an axis the exporter leaves as a `Reshape`, which
TensorRT 11's stricter parser rejects) and, for FP16, exporting with `--fp16` despite the tool's stale warning
(TensorRT >= 10 is strongly typed: precision comes from the graph's dtypes). ONNX Runtime's TensorRT execution
provider wants their pinned `libnvinfer.so.10`, so engines were built and validated with `trtexec` directly.

## More sparse and semi-dense models

```python
model = flashim.load("superpoint-lightglue", "bf16", overrides=["components.extractor.preset=bf16"])
m = model.match("a.jpg", "b.jpg", resize=1600, top_k=4096)      # SparseMatchResult
model = flashim.load("dedode-g", "int8"); model.extract("a.jpg", resize=1024, top_k=8192)   # SparseFeatures
model = flashim.load("loftr", "bf16"); model.match("a.jpg", "b.jpg", resize=840)            # pair-level: no extract()
```

Ten models were added, all as native PyTorch ports that load the released checkpoints with `strict=True` and have a parity
test against the upstream code (`tests/parity/<model>`; the upstream reference code is vendored in `tests/oracles`, DISK / LoFTR also
against kornia). Detector, descriptor and matching follow each method's own protocol:

| model | what it is | matching | working resolution |
|---|---|---|---|
| `superpoint` | VGG encoder, cell-softmax heatmap, 256-d | mutual NN, or LightGlue (`superpoint-lightglue`) | 1600 px |
| `disk` | U-Net of 5x5 convs (InstanceNorm + PReLU), 128-d | mutual NN, or LightGlue (`disk-lightglue`) | 1600 px |
| `alike` | ALIKED's plain-conv predecessor, variants `alike-t/s/n/l` | mutual NN | 1600 px |
| `xfeat` | 0.7M-parameter net, 64-d | mutual NN (0.82), or LighterGlue (`xfeat-lightglue`) | 1600 px |
| `dedode` / `dedode-g` | DeDoDe v2 detector L + descriptor B / G (VGG19 + DINOv2 ViT-L/14) | dual softmax | 1024 px |
| `dad` | DaD detector (VGG11) + DeDoDe-G descriptors | dual softmax | 1024 px |
| `ripe` | DeDoDe-style heatmap + hypercolumn 256-d descriptors | mutual NN | 1024 px |
| `loftr` | detector-free: ResNetFPN, linear-attention transformer, coarse + fine matching | built in | 840 px |
| `sift`, `orb` | OpenCV, CPU detection, GPU ratio-test matching; fp32 only | mutual NN + ratio 0.8 | 1600 px |

Presets: `fp32` (checkpoint precision, exact BatchNorm fold), `bf16` (trunk and heads in BF16; keypoint selection, softmaxes,
sampling, the LoFTR attention core and coarse similarity stay FP32), `int8` (static symmetric per-channel weights, one
calibrated per-tensor activation scale) and `fp8` (static MXFP8) on the convolutions and linears listed in each
`spec.py`. Convolutions run as im2col + GEMM (`quant/methods/conv2d.py`: quantize on NHWC, gather the *quantized bytes*
with `ops.im2col_gather` — one Triton kernel per conv call, see below — into the existing INT8 / MXFP8 GEMM; any kernel /
stride / padding, `groups == 1`). Calibration: `scripts/calibrate_extractor.py` (24 to 32 pairs of the MegaDepth
calibration split, disjoint from MegaDepth-1500) and, for the LightGlue pairings, `scripts/calibrate_lightglue.py
--extractor <model>`; tables ship next to each model.

`extract()` latency, RTX 5070 Ti, eager, one MegaDepth image at the protocol resolution above with 4096 keypoints (8192 for
DeDoDe / DaD; LoFTR: `match` of a pair). Ratios are over the model's own `bf16` row (`benchmarks/speed/sparse_matrix.sh`):

| model | fp32 | bf16 | int8 | fp8 | int8 / bf16 | fp8 / bf16 |
|---|---|---|---|---|---|---|
| superpoint | 25.4 | 14.1 | 18.2 | 20.0 | 0.77x | 0.70x |
| disk | 69.6 | 64.5 | 91.2 | 80.8 | 0.71x | 0.80x |
| alike | 14.4 | 11.9 | 18.8 | 24.4 | 0.63x | 0.49x |
| xfeat | 3.6 | 3.5 | 4.0 | 5.5 | 0.89x | 0.63x |
| dedode | 132.2 | 90.3 | 67.0 | 91.6 | **1.35x** | 0.99x |
| dedode-g | 361.6 | 130.8 | 107.0 | 117.3 | **1.22x** | **1.12x** |
| dad | 288.9 | 96.9 | 87.1 | 88.6 | **1.11x** | **1.09x** |
| ripe | 78.5 | 53.7 | 35.8 | 51.2 | **1.50x** | **1.05x** |
| loftr | 71.2 | 46.9 | 44.7 | 52.0 | **1.05x** | 0.90x |
| sift / orb | 34.6 / 13.5 | | | | | |

BF16 is the biggest win: 1.0x to 1.6x over fp32 for the CNNs and 2.6x for the DeDoDe-G / DaD stack (DINOv2 ViT-L + refiners).

**Two bugs made INT8/FP8 slower than BF16 on most models; both are fixed, and FP8 now matches or beats BF16 on four of
nine.**

1. **The im2col gather, not the GEMM, was the first bottleneck.** Profiling (`benchmarks/profile/extractor_profile.py`,
   SuperPoint `int8`, 1600 px) showed the INT8 GEMMs taking 3.05 ms for the nine quantized convolutions (about what
   BF16's cuDNN convolutions take for the same layers), but the `F.pad` + `torch.cat` that built the im2col matrix
   taking 14.5 ms — 47% of all kernel time, more than the GEMM plus the convolutions it replaced. `F.pad` materialized
   a whole padded copy of the activation, and `torch.cat` then copied each of the `kh*kw` strided taps into the
   im2col buffer separately: `kh*kw + 1` extra full passes over the data for a memory-bound 1-byte gather. Fixed with
   `ops.im2col_gather`, one Triton kernel (`backends/triton/conv.py`) that computes the padding bounds inline and
   writes every tap of an output row in one program, no padded intermediate or per-tap launches: SuperPoint's im2col
   dropped from 14.5 ms to 6.65 ms (bit-identical — `tests/quant/test_conv2d.py::test_im2col_matches_unfold` checks it
   against `F.unfold`).
2. **MXFP8's own activation quantization was calling the wrong function.** `StaticMXFP8Linear.forward` (and the new
   `StaticMXFP8Conv2d`, which copied it) imported `quantize_static` straight from `flashim.quant.tensor` — the plain
   torch reference (pad to a multiple of 32, reshape into blocks, divide by the exponent, clamp, cast: five separate
   kernel launches) — instead of calling through `ops.quantize_mxfp8`, the dispatched op whose Triton backend
   (`backends/triton/quantize.py`) already does the same thing in one fused kernel. That backend exists and is tested
   (`tests/ops/test_ops_cuda.py::test_quantize_mxfp8`) but nothing on this call path used it. Fixed by calling
   `ops.quantize_mxfp8` in both classes; this also speeds up any other model's plain (non-fused-producer) MXFP8
   linears, not just the new conv layers. Effect (fp8, on top of fix 1): 15% to 39% faster; `dedode-g`, `dad` and
   `ripe` now beat `bf16`, and `dedode` is within 1%.

**What's left, and why it's not worth chasing further:** cuBLAS's block-scaled MXFP8 GEMM needs both the padded `N`
and `K` (`kh*kw*C_in`, rounded up to 32) divisible by 128 (`backends/torch_native/gemm.py:_cublas_shapes`); anything
else falls back to a Triton FP8 GEMM that is 7x to 9x slower per call (0.08 ms vs 0.73 ms in the SuperPoint profile).
`model.quant_report.summary()["linear_kernels"]` shows the split per model: DeDoDe / DaD / RIPE / LoFTR are already
80-90%+ cuBLAS (their linears and 1x1 refiner convs sit at 256+ channels), which is why fixes 1 and 2 alone got them
to parity or a win. DISK, ALIKE and XFeat are the opposite — 0/7, 1/14 and 2/19 cuBLAS — because their conv channel
counts (16 to 64) never reach 128; that's architectural, not a bug, and padding those channels up to 128 to buy
cuBLAS eligibility would multiply their real FLOPs and gather bytes by 2x to 8x for a GEMM that's already cheap
relative to the gather, a bad trade. Not attempted for those. A true implicit-GEMM kernel (taps gathered inside
the GEMM's shared-memory load, no im2col buffer, bias/ReLU/requantize in the epilogue) would help every model and
was not attempted either — the biggest remaining project here.

**A custom Triton/TileLang GEMM to remove the 128-alignment need directly, for DISK and LoFTR (the two models with
some genuinely large `K`, up to 3200): tried, and it helps a little, once the design landed on the right shape.**
MXFP8's own scale-block size (32) equals the K-width of this GPU's FP8 tensor-core instruction (`mma.sync.m16n8k32`
— available here: `Feature.MMA_SYNC` + `Feature.FP8_E4M3`, confirmed no `Feature.TCGEN05` needed, since cuBLAS's own
kernel doesn't have it either on this consumer-Blackwell card), so a kernel can rescale-and-accumulate per 32-wide
block with no alignment requirement at all. Two false starts before it worked:
- *v1* (one small MMA + one global-memory round trip per 32-wide block): 3x to 16x **slower** than the Triton
  fallback at every shape tried — reverted immediately, nothing shipped.
- *v2* (load the entire `K` into shared memory once per tile, to remove those round trips): correct and fast in
  isolation, but this GPU's opt-in shared memory per block is only ~99KB (`shared_memory_per_block_optin`) — far
  short of datacenter Blackwell's ~227KB — so it doesn't fit for the actual `K` values here (1152-3200) at a useful
  tile size.
- *v3* (shipped, `backends/tilelang/{gemm.py,_gemm_kernels.py}`): batch `K` in chunks (a divisor of `K`, sized to
  fit the shared-memory budget) instead of the full width or single 32-wide blocks — one shared-memory load per
  chunk, several MMAs + a batched rescale-accumulate from data already resident. Registered as `ops.mxfp8_linear`'s
  `"tilelang"` impl, gated to `K >= 2000` and cuBLAS-ineligible (`backends/tilelang/gemm.py:MIN_K`): below that,
  measured roughly break-even with Triton (0.94x-1.01x) or worse; above it, 1.1x-2.5x faster **in isolation**.
  End-to-end, after also fixing a real bug the integration surfaced (padding the activation's row count up to the
  tile size with `F.pad` looked free — at most 63 extra rows — but pad allocates and copies the *whole* tensor, an
  `O(M*K)` pass that was as expensive as the GEMM itself for LoFTR's ~130k-row activations; fixed by splitting into
  a zero-copy "bulk" launch on the exact multiple of the tile size plus one small launch for the remainder): DISK's
  `extract()` is a real, repeatable ~4% faster (best-of-3: 80.8 -> 77.4-77.6 ms); LoFTR is neutral (best-of-3:
  52.3-52.8 ms, matching its pre-tilelang 52.0 ms). The isolated GEMM win doesn't show up end-to-end because the
  im2col gather, activation quantization and everything else each layer still needs dominate the total. Verified
  bit-identical (`tests/ops/test_ops_cuda.py`, `tests/parity/{disk,loftr}`); every other model's quantized layers
  have `K <= 576` and are untouched (still Triton/cuBLAS as before).

Fidelity to each model's own fp32 output on 20 held-out MegaDepth-1500 pairs (`benchmarks/accuracy/fidelity.py`, full table
in `results/sparse_models.md`): `bf16` keeps 92 to 99% of the keypoints (within 1.5 px) and descriptor cosine >= 0.994; `int8`
keeps cosine >= 0.99 on every model and 87 to 97% of the keypoints; `fp8` is clearly the least faithful on the small nets
(SuperPoint: 70% of the keypoints, descriptor cosine 0.953, 18% fewer matches; XFeat: 73% of the keypoints), fine on
DISK / DeDoDe / RIPE (cosine >= 0.986). **INT8 on DINOv2 is broken**: quantizing the ViT-L linears of DeDoDe-G / DaD with one
per-tensor scale gives a dense-descriptor cosine of 0.79 (min 0.01); attention and the MLP down-projection each break it
(DINOv2's outlier activations). The `int8` presets of `dedode-g` and `dad` therefore leave the ViT in BF16 (cosine 0.997);
their `fp8` presets keep it in MXFP8 (per-32-block scales: cosine 0.994).

Task-level accuracy (MegaDepth-1500 AUC, RDD's protocol) of the quantized presets is **not measured yet**: it is a 3 to 5
hour GPU job (`benchmarks/accuracy/sparse_matrix.sh`, resumable, then `benchmarks/report_sparse.py`). Until then only the
fidelity proxy above backs the `int8` / `fp8` presets. Sanity runs (fp32, first 100 pairs, AUC@5): LoFTR 0.52, DeDoDe-G 0.51
(0.55 at 1024 px), SuperPoint+LightGlue 0.50 (full 1500 pairs 0.503), DeDoDe 0.50, DaD 0.46 (0.48 at 1024 px), DISK+LightGlue
0.45, XFeat+LighterGlue 0.46, ALIKE 0.45, SIFT 0.33, RIPE 0.37, SuperPoint with plain mutual NN 0.28, ORB 0.14.

Not done: SIFT + LightGlue (needs the scale / orientation position encoding), an ALIKE + LightGlue pairing (no released
weights), compiled engines (`engine.kind=compile` is wired up for every model but not benchmarked), and BF16 refiner-block
fusion for DeDoDe / DaD / RIPE (their depthwise 5x5 + ReLU + 1x1 blocks are exactly RoMaV2's `SeparableConvBlock` pattern:
the fused `depthwise5x5_relu_pointwise` kernels would apply after the BatchNorm fold; the depthwise convs are 36% of DeDoDe's
BF16 kernel time).

## Layout

```
flashim/
  platforms/   vendor-aware device detection: dispatch key, arch, ISA features (tma, wgmma, tcgen05, fp8,
               mxfp8 scaled_mm, ...), toolchains (triton, tilelang, nvcc); pluggable per device type
  ops/         op declarations: signature + torch reference (the correctness oracle and fallback);
               files by computation (attention, conv, correlation, density, gemm, quantize);
               model-specific ops in a file named after the model (romav2.py, aliked.py)
  backends/    optimized impls, one package per kernel language / library, mirroring ops/ file names:
               triton/, tilelang/, cuda/ (JIT CUDA C++), torch_native/ (cuBLAS), external/ (sageattention);
               out-of-tree backends via the `flashim.backends` entry point
  log.py       logging (FLASHIM_LOG), dispatch tracing, profiler / NVTX ranges
  config.py    RunConfig: typed, layered (preset < file < overrides), serializable run configuration
  build.py     ModelSpec (presets, passes, engines) + the generic builder all models go through
  layers/      building blocks the passes recognize by type (TransformerBlock, SeparableConvBlock,
               DeformableEncoder, ...): a model built from them gets quantization + fused kernels without
               model-specific code
  quant/       formats (MX), QuantizedTensor, schemes (compressed-tensors vocabulary), calibration,
               methods (StaticMXFP8Linear = ops.mxfp8_linear bound to its weight: cuBLAS MXFP8 > Triton FP8 >
               emulated reference, via normal op dispatch), fusion passes
  engines/     region torch.compile, experimental CUDA-graph runner
  models/      romav2 (DINOv3 ViT-L/16, MV-ViT, DPT head, conv refiners, presets, calibration)
               rdd (ConvNeXt, deformable encoder, detector, sparse keypoints / matching, presets, calibration)
               aliked (deformable-conv encoder, SDDH descriptor head, DKD keypoints / matching, presets)
               superpoint, alike, disk, xfeat, dedode (+ dad, ripe share its blocks), loftr, sift, orb (see above);
               sparse.py / sparse_spec.py: base class + spec factory shared by the sparse extractors
  geometry/    pose estimation + AUC metrics
benchmarks/    speed/forward.py, accuracy/run.py (MegaDepth-1500, ScanNet-1500), accuracy/run_sparse.py (RDD)
scripts/       calibrate_romav2.py, calibrate_rdd.py
tests/         unit, ops (impl vs reference), quant, parity (vs the vendored upstream code in tests/oracles)
```

Op overrides: `FLASHIM_OPS="rope_qk:reference,-local_corr"`; `FLASHIM_STRICT_OPS=1` raises
if a GPU call falls back to a torch reference.

## Platforms

FP8 presets run natively on Blackwell (sm_100 / sm_120) through `F.scaled_mm`
block-scaled MXFP8, and through a Triton FP8 GEMM on sm_89 / sm_90.

A GPU without FP8 (Ampere, Jetson Orin) builds the preset's INT8 counterpart instead: `fp8` -> `int8`; RDD, which has no INT8 preset, ->
`bf16`. A warning says so, and `model.preset` / `model.build_report.fallbacks` record it. On a Jetson AGX Orin
(eager) RoMaV2 `fp8` would take 3.3 s per pair emulated, against 1.1 s for the earlier `int8-full` (now `int8`, which adds the fast refiner kernels) and 2.0 s for `bf16`.

FP8 is kept, on emulation kernels (same numerics contract, slower), on the CPU, when a calibration table is
passed with the preset, and by `flashim.build_config(flashim.resolve(name, preset))`; `optimize.strict=true`
raises instead of replacing or emulating.

## Checkpoints

Every model downloads its upstream release on first use into the torch hub checkpoint folder
(`$TORCH_HOME/hub/checkpoints`, default `~/.cache/torch/hub/checkpoints`) and checks it against the SHA-256 pinned
in `flashim/models/<model>/weights.py`; a file already there is used as is. `checkpoint=` or the model's
`FLASHIM_<MODEL>_CHECKPOINT` variable (e.g. `FLASHIM_RDD_CHECKPOINT`) points at a local file instead. For an
offline machine, copy the checkpoint folder over. Calibration tables ship in the repository.

## Checkpoints

Every model downloads its upstream release on first use into the torch hub checkpoint folder
(`$TORCH_HOME/hub/checkpoints`, default `~/.cache/torch/hub/checkpoints`) and checks it against the SHA-256 pinned
in `flashim/models/<model>/weights.py`; a file already there is used as is. `checkpoint=` or the model's
`FLASHIM_<MODEL>_CHECKPOINT` variable (e.g. `FLASHIM_RDD_CHECKPOINT`) points at a local file instead. For an
offline machine, copy the checkpoint folder over. Calibration tables ship in the repository.

## Development

```bash
uv sync --extra eval                     # Python 3.12 (.python-version), torch 2.11 (cu130), triton 3.6
uv run pytest -m "not slow"              # unit / ops / quant
uv run pytest tests/parity               # bit-exactness vs the vendored upstream code (needs weights + GPU)
uv run python benchmarks/speed/forward.py --preset fp8 --engine compile
uv run python benchmarks/accuracy/run.py --dataset mega1500 --preset fp8 --seed 17
```

The DINOv3 weights inside the RoMaV2 checkpoint are covered by the DINOv3 License.
