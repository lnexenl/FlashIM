# LoMa

[LoMa](https://github.com/davnords/LoMa) (ECCV 2026) as a native port: DaD keypoints (VGG11), LoMa-trained DeDoDe
descriptors and a LightGlue-style matcher trained on them. Five releases, one checkpoint each (matcher + frozen detector
+ descriptor):

| pipeline | matcher (9 layers) | descriptor | checkpoint |
|---|---|---|---|
| `loma-b128` | 128 -> 256, 4 heads | DeDoDe-B (VGG19), 128-d | `loma_B128.pth` (127 MB) |
| `loma-b` (= `loma`) | 256, 4 heads | DeDoDe-G (VGG19 + DINOv2 ViT-L/14), 256-d | `loma_B.pt` (758 MB) |
| `loma-l` | 256 -> 512, 8 heads | as `loma-b` (bit-equal) | `loma_L.pth` (900 MB) |
| `loma-g` | 256 -> 1024, 16 heads | as `loma-b` (bit-equal) | `loma_G.pth` (1.47 GB) |
| `loma-r` | as `loma-b`, rotation-augmented | its own DeDoDe-G | `loma_R.pth` (758 MB) |

```python
model = flashim.load("loma-g")                              # bf16
m = model.match("a.jpg", "b.jpg", max_matches=4096)         # keypoints per image (default 2048); resize: DaD's long side (1024)
```

The components are `loma-extractor` and `loma-matcher` (config key `variant`). As upstream's path API: DaD on the image
resized to a 1024 px long side (sides floored to multiples of 8), the descriptor on a separate 784 x 784 resize without
ImageNet normalization, both PIL-bicubic resizes reproduced in torch; descriptors are not L2-normalized; the matcher
sees DaD's normalized coordinates and scores the last layer's dual softmax (mutual maxima above 0.1). Keypoints are
FlashIM's pixel centers (upstream's are pixel edges: `x_upstream = x + 0.5`).

## Presets

`fp32` (reference), **`bf16`** (default; upstream also runs BF16 autocast by default), `int8` and `fp8` (both
components quantized: the matcher's transformer linears as LightGlue's presets, the extractor's VGG / refiner
convolutions and the DINOv2 linears: per-token dynamic INT8 in `int8` as `dedode-g`'s, static MXFP8 in `fp8` as DaD's). Calibration tables: matcher per variant, extractor per
descriptor weight set (`b`/`l`/`g` share one), 32 MegaDepth calibration pairs at 1024 px / 4096 keypoints.

The matcher's `int8` preset folds Hadamard rotations into the weights first (pass `rotate_residual_stream`: the
residual stream has no LayerNorm, so the basis change is free) and smooths the one input it cannot reach, `ffn.3`'s,
with per-channel SmoothQuant factors applied in the input quantization; scales stay static per tensor. On real kernels
this raised match agreement with the bf16 matcher from 0.92-0.95 to 0.95-0.97 and put every variant within the pose-AUC
gate on MegaDepth-1500 (ΔAUC@5 vs `bf16` +0.01 to -0.42, 10 RANSAC seeds, paired), for ~1.5 % of the matcher's time.

On sm_89 / sm_90 / sm_120, `int8` also runs DeDoDe-G's int8 passes in the extractor (decoder and stem fusions, fused ViT glue,
patch-embedding GEMM, SageAttention on ViT blocks 2-23; `loma-l` keeps the bit-exact stem and blocks 4-23) and three in
the matcher: `sage_attention`, `fused_int8_glue` (one residual stream for both images, quantize / RoPE / image swap /
LayerNorm + GELU fused into the GEMMs and their producers; bit-identical) and `cuda_graph` (one CUDA graph per
keypoint-count pair, eager engine; bit-identical). On `sm_120`, `fp8` also enables DINOv2 SageAttention, matcher
`fused_bias`, `cuda_graph` (with a `sdpa_graph_safe` guard), and `sage-fp8` on `dim >= 512`; `loma-b128 fp8` uses
`ModelSpec.quant_configs` per-variant `<variant>/<quant>` lookup (`b128/fp8`) to keep `*.decoder.*` and matcher
`transformers.{4,5}.*` in BF16 (recovering ScanNet-1500 AUC@5 from `-0.0154` on `dev` to `-0.0008`).

Measured eager latencies:

- **L20 (`sm_89`) and H20 (`sm_90`)**, full pair (2 extracts from JPEG at 1024 px, 4096 keypoints, + match, `eager`):
  L20 `int8` 1.68-1.77x and H20 `int8` 1.53-1.65x faster than `bf16` (e.g. `loma-l` L20 138.9 -> 82.5 ms, H20
  118.1 -> 77.0 ms), MegaDepth-1500 ΔAUC@5 vs `bf16` -0.30 to -0.10 (L20) and -0.09 to +0.23 (H20) over the five
  variants (OpenCV RANSAC 0.5 px, 10 seeds, paired).
- **RTX 5070 Ti (`sm_120`, `eager`)**: `loma-b128` pair at 1024 px (`fp8` vs `bf16`): `40.11` vs `51.17 ms` at 4096 kp
  (1.28x), `33.40` vs `44.19 ms` at 2048 kp (1.32x); matchers before -> after the `sm_120` sweep: `loma-b fp8` (2048 kp)
  `7.21 -> 3.52 ms` (2.05x), `loma-l` (4096 kp) `int8` `20.32 -> 12.97 ms` (1.57x) / `fp8` `26.22 -> 18.61 ms` (1.41x),
  `loma-g` (4096 kp) `int8` `43.57 -> 29.25 ms` (1.49x) / `fp8` `65.58 -> 39.30 ms` (1.67x).

Without the graph the 256-dim INT8 matcher is launch-bound and slower than BF16. The graph needs SageAttention launching
on the current stream; otherwise the capture fails and the matcher runs eagerly (warned).

Fidelity against `fp32` on 20 MegaDepth-1500 pairs (`benchmarks/accuracy/fidelity.py`, 1024 px, 4096 keypoints, RTX 5070 Ti
`sm_120`, `eager`), the fraction of `fp32` matches kept: `bf16` 0.912-0.917, `int8` 0.857-0.871 (before the rotation),
`fp8` 0.800-0.843. Full MegaDepth-1500 and ScanNet-1500 pose AUC@5 gate results on `sm_120` (`0 FAIL` across all five
variants and all presets) are in [benchmarks.md](../benchmarks.md#accuracy-of-the-pre-2026-11-models-2026-11-gate-sm_120)
and [platforms.md](../platforms.md).

## Licenses

LoMa is MIT except its matcher, derived from LightGlue (Apache-2.0); DaD / DeDoDe MIT; DINOv2 (inside every checkpoint
but B128) Apache-2.0. FlashIM's port is under FlashIM's license.
