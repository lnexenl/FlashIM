# FlashIM

Fast, quantized, multi-platform inference for image-matching models.

```python
import flashim

model = flashim.load("romav2", preset="fp8", setting="mega1500", engine="compile")
result = model.match("a.jpg", "b.jpg")                 # DenseMatchResult (warp, overlap, precision)
matches, conf, prec_a, prec_b = model.sample(result, 5000)
kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
```

## Models

| model | presets | notes |
|---|---|---|
| `romav2` | `bf16`, `fp8-p8`, `fp8-s4`, `fp8` (default), `fp8-fast` | native port of RoMa v2; `bf16`/`fp8-*` outputs are bit-identical to the RoMaV2 reference implementation |

RoMaV2 presets (MXFP8 = E4M3 values, E8M0 scale per 32 elements):

- `fp8-p8` — every DINOv3 / matcher-ViT linear with *static* calibrated activation scales, fused LayerNorm→quant and GELU→quant producers, BF16 SDPA attention, BF16 refiners.
- `fp8-s4` / `fp8` — additionally fuse stride-4 (and stride-2) refiner blocks: one Triton kernel for depthwise 5x5 + bias + ReLU + *dynamic* MXFP8 quantization, then a block-scaled MXFP8 pointwise GEMM.
- `fp8-fast` — `fp8` plus FlashIM refiner kernels: fused BF16 stride-1 conv blocks, a fused refiner
  input (warp sampling, displacement embedding and local correlation in one pass) and fused FP32
  output heads. Not bit-identical to RoMaV2 (reduction order / rounding points); validated by pose AUC.

Engines: `eager`; `compile` (DINOv3 + matcher ViT, RoMaV2's validated setting); `compile-full`
(also VGG with folded BatchNorm, the DPT head and the refiner conv stacks).

Measured on an RTX 5070 Ti (mega1500 setting, 800² LR + 1024² HR, bidirectional, B=1, preloaded forward):

| configuration | forward | vs BF16 |
|---|---|---|
| `bf16` + `compile` | 344 ms | 1.0x |
| RoMaV2 FastFP8 (s4+s2 fused) + compile | 210 ms | 1.64x |
| `fp8` + `compile` | 204 ms | 1.69x |
| `fp8` + `compile-full` | 165 ms | 2.09x |
| `fp8-fast` + `compile-full` | 110 ms | 3.13x |

Pose AUC@5 change vs `bf16` (3 seeds): `fp8`+`compile` Mega1500 -0.0042 / ScanNet1500 -0.0017;
`fp8`+`compile-full` -0.0020 / -0.0026; `fp8-fast`+`compile-full` -0.0023 / +0.0007
(gate: 3-seed mean >= -0.005; every `fp8-fast` seed is within -0.005).

`sample()` (KDE-balanced match sampling) takes 0.86 ms (RoMaV2: 16 ms, 1.5 GiB scratch); pose AUC
vs RoMaV2's sampling +0.0012 / -0.0010 (3 seeds; `FLASHIM_FAST_SAMPLING=0` restores it); `match(..., num_samples=N)` returns pixel keypoints.

Coordinate heads, the patch embedding, refiner projections and displacement
embeddings are protected: a quantization config that targets them is rejected.

## Layout

```
flashim/
  platforms/   device + feature detection (triton, fp8, mxfp8 scaled_mm, ...)
  ops/         every op = torch reference + optional per-platform impls (Triton, cuBLAS, fused_local_corr)
  quant/       formats (MX), QuantizedTensor, schemes (compressed-tensors vocabulary), calibration,
               methods (StaticMXFP8Linear), kernels (cuBLAS MXFP8 > Triton FP8 > emulation), fusion passes
  engines/     region torch.compile, experimental CUDA-graph runner
  models/      romav2 (DINOv3 ViT-L/16, MV-ViT, DPT head, conv refiners, presets, calibration)
  geometry/    pose estimation + AUC metrics
benchmarks/    speed/forward.py, accuracy/run.py (MegaDepth-1500, ScanNet-1500)
scripts/       calibrate_romav2.py
tests/         unit, ops (impl vs reference), quant, parity (vs the RoMaV2 package)
```

Op overrides: `FLASHIM_OPS="rope_qk:reference,-local_corr"`; `FLASHIM_STRICT_OPS=1` raises
if a GPU call falls back to a torch reference.

## Platforms

FP8 presets run natively on Blackwell (sm_100 / sm_120) through `F.scaled_mm`
block-scaled MXFP8. Elsewhere they fall back to a Triton FP8 GEMM (sm_89+) or to
emulation kernels (any device, including CPU) — same numerics contract, slower.

## Development

```bash
uv sync --extra eval                     # torch 2.11 (cu130), triton 3.6, fused-local-corr
uv run pytest -m "not slow"              # unit / ops / quant
uv run pytest tests/parity               # bit-exactness vs ../RoMaV2 (needs weights + GPU)
uv run python benchmarks/speed/forward.py --preset fp8 --engine compile
uv run python benchmarks/accuracy/run.py --dataset mega1500 --preset fp8 --seed 17
```

The DINOv3 weights inside the RoMaV2 checkpoint are covered by the DINOv3 License.
