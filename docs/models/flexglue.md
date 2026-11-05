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

`match_keypoints` at 2048 RDD keypoints and `match()` on a MegaDepth pair, RTX 5070 Ti, best of interleaved runs:

| configuration | `match_keypoints` eager / compile | `match()` eager / compile | MegaDepth-1500 AUC@5 (vs bf16) |
|---|---|---|---|
| `bf16` | 42.7 / 32.5 ms | 47.2 / 36.6 ms | 0.6397 |
| `fp8` | 37.7 / 27.5 ms | 42.2 / 31.3 ms | 0.6369 (-0.0028) |
| `int8` | 36.4 / 27.5 ms | 41.0 / 31.4 ms | 0.6370 (-0.0027) |
| `fp8` + `optimize.attention=sage-fp8-fast` | 30.2 / 19.4 ms | 34.7 / 23.5 ms | 0.6371 (-0.0026) |
| `int8` + `optimize.attention=sage` | 33.5 / 24.9 ms | 38.3 / 28.9 ms | 0.6380 (-0.0017) |

Each preset is paired with the attention that matches the GPUs it targets: `int8` is the preset for GPUs without
FP8 (Ampere, Orin; `fp8` falls back to it there), so it uses `sage` (INT8 QK + FP16 PV, sm_80+), while the
`sage-fp8*` variants need FP8 tensor cores (sm_89+) like the `fp8` preset.

AUC: dense `match()`, 5000 matches, poselib 1.0px, mean of RANSAC seeds 17 / 29 / 42 (eager);
`benchmarks/accuracy/flexglue_matrix.sh`. Quantizing the descriptor costs MXFP8 0.0031 / INT8 0.0010-0.0019
AUC@5 (the transformer-only presets: 0.6388 / 0.6380) for ~1.6 ms compiled. `engine="compile"` is the larger win (it fuses the
per-head LayerNorms, the rotary and the residual adds around the linears; the first call compiles for tens of
seconds). Attention over the 64x64 = 4096 tokens is the largest remaining kernel (37% of eager fp8 before
SageAttention); `optimize.attention` swaps it for SageAttention (`sage`: INT8 QK + FP16 PV, `sage-fp8`: + FP8 PV,
`sage-fp8-fast`: + SageAttention2++'s FP16 PV accumulation). It is opt-in because its AUC drop, although inside the
0.005 gate, is measurable (transformer-only fp8: sage -0.0025, sage-fp8 -0.0022, sage-fp8-fast -0.0016 vs
-0.0009); it needs a BF16 preset (`fp32` rejects it). SageAttention3's NVFP4 attention measured 0.5475 (-0.092).

```python
m = flashim.load("flexglue", "fp8", engine="compile", overrides={"optimize.attention": "sage-fp8-fast"})
```

## Presets

| preset | what it does |
|---|---|
| `fp32` | the checkpoint's own precision |
| `bf16` | descriptor and transformer in BF16; the correspondence heads, search map, positions, entmax readout and mutual-NN stay FP32 |
| `int8` | `bf16` + the transformer and descriptor-MLP linears as static symmetric INT8 |
| `fp8` (default) | `bf16` + the transformer linears as static MXFP8 and the descriptor-MLP linears as static INT8 |

## When to use FlexGlue

FlexGlue is a matcher, not a detector: pair it with an extractor for keypoint matching, or dense-sample with
`match()` for a lower-cost detector-free warp. For pure sparse matching see [LightGlue](lightglue.md); for a
full dense matcher see [RoMaV2](romav2.md).
