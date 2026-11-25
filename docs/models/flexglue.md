# FlexGlue

Sparse-to-dense matcher: a native port of [FlexGlue](https://github.com/cvg/FlexGlue) (MIT). Unlike
[LightGlue](lightglue.md), which exchanges one vector per keypoint, FlexGlue describes each image densely with a
DINOv3-ConvNeXt backbone into a stride-8 search map, then runs 9 transformer layers over the search-map tokens.
Any query point's correspondence in the other image is regressed directly from that map by a 1.5-entmax readout
of the search cells plus a learned sub-cell offset head. Every transformer layer carries its own correspondence
head, together with a covisibility ("disambiguation") logit, so the network can stop decoding early on a pair it
judges confidently non-covisible.

## Usage

Keypoint matching: give pixels `[K, 2]` detected by an extractor (e.g. [RDD](rdd.md), ALIKED or RaCo):

```python
m = flashim.load("flexglue", "fp8", device="cuda")
matches = m.match_keypoints(img_a, kpts_a, img_b, kpts_b, threshold=0.1)   # Matches
```

Dense sampling: up to `max_matches` KDE-balanced correspondences drawn from the dense warp of both images, gated
like upstream's `gate_dense` (covisibility, confidence `threshold`, image borders, and a `max_distance_px` cycle
check through the reverse warp; the balancing is RoMaV2's):

```python
matches = m.match(img_a, img_b, max_matches=5000)                          # Matches
```

`match_features(SparseFeatures, ...)` is **not** supported: the matcher reads the dense search maps, so the
images themselves are needed. Use `match_keypoints` (images + keypoints) or `match` (dense sampling).

## Early stopping

Off by default (`arch.early_stop_decay=None` runs all 9 layers deterministically, like upstream). Enable it with

```python
m = flashim.load("flexglue", "fp8", overrides={"arch.early_stop_decay": 0.9, "arch.min_covisibility": 0.1})
```

The gate fires only on a confident non-pair: after layer `k`, if the batch's covisibility score has decayed below
`min_covisibility * (1-decay)**(9-k)`, decoding stops and layer `k`'s head is used (all gates then reject every
match). Covisible pairs keep running to the last layer, so the settings above rarely trigger (they never did on four
measured MegaDepth pairs and non-pairs), and `min_covisibility=0.1` also drops every match of a genuine but hard
pair whose covisibility is below it (one measured MegaDepth pair scores 0.086). The win is skipping the later layers on image pairs that do not overlap, at the cost of a host sync per checked layer.

## Resolution

The descriptor runs at a fixed 512x512 (`arch.describe_resize`): images are squashed to that frame (the aspect
ratio is not kept). Keypoints are given and returned in the original images' pixels, but the match gates
(mutual-NN distances, `max_distance_px`, the border checks) operate in the squashed 512x512 frame — so
`max_distance_px=3` tolerates more vertical drift on a tall image than a wide one. `match()` takes no `resize`.

## Quantization

The `int8` / `fp8` presets quantize 153 linears: the 9 transformer layers' QKV / output / FFN linears (81, self-
and cross-attention) and the pointwise MLP of every ConvNeXt block in the descriptor's backbone (72, `pwconv1` /
`pwconv2`: 83.7M of its 89.6M parameters). `int8` makes all of them static symmetric INT8 (per-output-channel
weights, one activation scale per tensor). `fp8` mixes formats: the transformer linears as static MXFP8 (E4M3
values, E8M0 block-32 scales), the descriptor's as the same INT8 (every GPU with FP8 tensor cores also has INT8
ones). Each part gets the format it loses least in: an MXFP8 descriptor costs 0.0012 AUC@5 more at the same speed. The backbone's convolutions, the descriptor's stride-8
head, the input projection, the Fourier position encoding and the correspondence heads are not quantized: the heads
feed the coordinate-sensitive entmax readout.

The calibration tables ship in `flashim/models/flexglue/calibration/`
(`flexglue_fp8_mega.safetensors`, `flexglue_int8_mega.safetensors`: 16 pairs of
`data/quantization/calibration_megadepth.jsonl`; `fp8` reads both); regenerate them with
`scripts/calibrate_flexglue.py`.

## Speed and accuracy

`match_keypoints` at 2048 RDD keypoints and dense `match()` (5000 KDE-sampled matches, 512x512 descriptor) on MegaDepth pair `505127/505128`, **RTX 5070 Ti (`sm_120`)**, torch `2.11.0+cu130`, `engine="eager"`, best of 2 interleaved rounds:

| preset | attention | `match_keypoints` | dense `match()` |
|---|---|---:|---:|
| `fp32` | torch SDPA | 145.1 ms | 146.3 ms |
| `bf16` | torch SDPA | 35.1 ms | 36.1 ms |
| `int8` | `sage` | 22.3 ms | 23.4 ms |
| `fp8` | `sage-fp8-fast` | **18.0 ms** | **19.0 ms** |

`engine="compile"` gains little (`bf16` 33.7 / 34.9 ms, `int8` 22.2 / 23.4 ms, `fp8` 17.9 / 19.1 ms; benchmark table).

The eager speed comes from six named passes in `spec.py` plus custom Triton kernels: `fold_convnext_gamma`, `channels_last` (NHWC ConvNeXt + head), `fuse_convnext_int8` (LN+quantize, GELU+quantize, INT8 residual epilogue), `fuse_transformer_glue` (MXFP8 cuBLAS fused bias, `ops.quantize_mxfp8_heads` / `ops.quantize_int8_heads`, `ops.layernorm_cat_quantize_{mxfp8,int8}`, FFN LN+GELU+quantize, shared `to_qk`/`to_v` quantize), `sage_attention`, and fixed-shape descriptor `cuda_graph`, together with Triton `ops.flexglue_entmax_readout` (zero-sort safeguarded Newton + active-set quadratic), `ops.qkv_norm_rope_interleaved`, `ops.qk_norm_swap_heads`, 3D `ops.kde_density`, border-padded `ops.bilinear_sample_points`, tiled 4D `ops.reciprocal_nn`, and dense `MatchHead.match_dense_pair` reuse. Whole-transformer (`decode_pair`) CUDA graphing was also tested with `sdpa_graph_safe()` on `sm_120`: unpatched SageAttention fails `sdpa_graph_safe()` (legacy default stream), and with graph-safe torch SDPA `decode_pair` is GPU-bound at 4096 tokens/image (`bf16` 25.71 -> 25.70 ms, 1.001x), so only the descriptor is graphed.

**Accuracy gate** (RTX 5070 Ti `sm_120`, `engine="eager"`, `FLASHIM_IMAGE_DECODER=pil`, dense `match()` at native 512x512, 5000 KDE-sampled matches, PoseLib LO-RANSAC 1.0 px, mean of seeds `17 / 29 / 42`, paired bootstrap 95% CI `B=10000` vs the unoptimized `bf16`; **5 PASS, 1 BORDERLINE, 0 FAIL**):

| dataset | preset | AUC@5 | AUC@10 | AUC@20 | Δ AUC@5 vs unoptimized `bf16` (95% CI) | gate |
|---|---|---:|---:|---:|---|---|
| MegaDepth-1500 | unoptimized `bf16` | 0.6399 | 0.7728 | 0.8653 | — | BASE |
| MegaDepth-1500 | `bf16` | 0.6394 | 0.7724 | 0.8653 | `-0.0005 [-0.0018, +0.0006]` | PASS |
| MegaDepth-1500 | `int8` | 0.6387 | 0.7727 | 0.8659 | `-0.0012 [-0.0044, +0.0020]` | PASS |
| MegaDepth-1500 | `fp8` | 0.6381 | 0.7715 | 0.8651 | `-0.0019 [-0.0057, +0.0021]` | BORDERLINE |
| ScanNet-1500 | unoptimized `bf16` | 0.3431 | 0.5712 | 0.7419 | — | BASE |
| ScanNet-1500 | `bf16` | 0.3429 | 0.5710 | 0.7420 | `-0.0002 [-0.0013, +0.0007]` | PASS |
| ScanNet-1500 | `int8` | 0.3420 | 0.5702 | 0.7414 | `-0.0010 [-0.0040, +0.0019]` | PASS |
| ScanNet-1500 | `fp8` | 0.3440 | 0.5709 | 0.7421 | `+0.0009 [-0.0023, +0.0041]` | PASS |

The `int8` / `fp8` presets run SageAttention (the `sage_attention` pass) on the architectures where it was measured
faster than torch SDPA, with the variant the preset's GPUs can run: `fp8` uses `sage-fp8-fast` (INT8 QK + FP8 PV) on
sm_89 / sm_120; `int8`, the preset for GPUs without FP8 (Ampere, Orin; `fp8` falls back to it there), uses `sage`
(INT8 QK + FP16 PV) on sm_80 / 86 / 87 / 89 / 120 (Ampere and Orin not timed). On Hopper they keep torch SDPA:
SageAttention's sm_89 kernels are slower than cuDNN SDPA on an H20 (compiled `match_keypoints` int8 31.0 -> 35.8,
fp8 31.7 -> 37.3 ms); its sm_90 kernel is faster (`optimize.attention=sage-fp8-sm90-warp`: 28.8 ms) but FlexGlue's
AUC with it is not measured. Without the `sageattention` package attention is torch SDPA (logged; the pass's entry in
`model.build_report.passes` says what ran). `optimize.attention` pins any impl instead; `optimize.passes=[]` turns
SageAttention off.

AUC: dense `match()`, 5000 matches, poselib 1.0px, mean of RANSAC seeds 17 / 29 / 42 (eager);
`benchmarks/accuracy/flexglue_matrix.sh`. Quantizing the descriptor costs MXFP8 0.0031 / INT8 0.0010-0.0019
AUC@5 (the transformer-only presets: 0.6388 / 0.6380) for ~1.6 ms compiled. `engine="compile"` is the larger win (it fuses the
per-head LayerNorms, the rotary and the residual adds around the linears; the first call compiles for tens of
seconds). Attention over the 64x64 = 4096 tokens is the largest kernel with torch SDPA (37% of eager fp8);
SageAttention variants: `sage` (INT8 QK + FP16 PV), `sage-fp8` (+ FP8 PV), `sage-fp8-fast` (+ SageAttention2++'s
FP16 PV accumulation). With a transformer-only fp8 they measured sage -0.0025, sage-fp8 -0.0022, sage-fp8-fast
-0.0016 vs -0.0009 with torch SDPA; `optimize.attention` needs a BF16 preset (`fp32` rejects it). SageAttention3's
NVFP4 attention measured 0.5475 (-0.092).

```python
m = flashim.load("flexglue", "fp8", engine="compile")                                  # sage-fp8-fast on sm_89 / sm_120
m = flashim.load("flexglue", "fp8", overrides={"optimize.attention": "sage-fp8-sm90-warp"})  # pin an impl
```

## Presets

| preset | what it does |
|---|---|
| `fp32` | the checkpoint's own precision |
| `bf16` | descriptor and transformer in BF16; the correspondence heads, search map, positions, entmax readout and mutual-NN stay FP32 |
| `int8` | `bf16` + the transformer and descriptor-MLP linears as static symmetric INT8; SageAttention `sage` |
| `fp8` (default) | `bf16` + the transformer linears as static MXFP8 and the descriptor-MLP linears as static INT8; SageAttention `sage-fp8-fast` |

## When to use FlexGlue

FlexGlue is a matcher, not a detector: pair it with an extractor for keypoint matching, or dense-sample with
`match()` for a lower-cost detector-free warp. For pure sparse matching see [LightGlue](lightglue.md); for a
full dense matcher see [RoMaV2](romav2.md).
