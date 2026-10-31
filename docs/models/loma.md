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
convolutions and, for `fp8`, the DINOv2 linears as DaD's). Calibration tables: matcher per variant, extractor per
descriptor weight set (`b`/`l`/`g` share one), 32 MegaDepth calibration pairs at 1024 px / 4096 keypoints.

The matcher's `int8` preset folds Hadamard rotations into the weights first (pass `rotate_residual_stream`: the
residual stream has no LayerNorm, so the basis change is free) and smooths the one input it cannot reach, `ffn.3`'s,
with per-channel SmoothQuant factors applied in the input quantization; scales stay static per tensor. On real kernels
this raised match agreement with the bf16 matcher from 0.92-0.95 to 0.95-0.97 and put every variant within the pose-AUC
gate on MegaDepth-1500 (ΔAUC@5 vs `bf16` +0.01 to -0.42, 10 RANSAC seeds, paired), for ~1.5 % of the matcher's time.

Fidelity against `fp32` on 20 MegaDepth-1500 pairs (`benchmarks/accuracy/fidelity.py`, 1024 px, 4096 keypoints), the
fraction of `fp32` matches kept: `bf16` 0.912-0.917, `int8` 0.857-0.871 (before the rotation), `fp8` 0.800-0.843.
MegaDepth-1500 pose AUC per preset and variant, and the commands, are in `results/loma/README.md`.

## Licenses

LoMa is MIT except its matcher, derived from LightGlue (Apache-2.0); DaD / DeDoDe MIT; DINOv2 (inside every checkpoint
but B128) Apache-2.0. FlashIM's port is under FlashIM's license.
