# LoMa speed experiments (prototypes, not library code)

Prototypes and probes from the LoMa INT8 / fusion work on the RTX 5070 Ti (sm_120). Nothing here is wired into
`flashim.load`: each script loads a preset and patches modules in place. Candidates for library Passes once they hold
up on other architectures. Run from the repo root with `PYTHONPATH=$PWD`; outputs go to the gitignored `results/loma/`.

| script | what it does |
|---|---|
| `vit_dynamic_int8.py` | per-token (per-row) dynamic INT8 for DeDoDe-G's DINOv2 ViT linears, which the int8 preset leaves BF16 (`dedode.spec.VIT_IGNORE`): Triton row quantize + INT8 GEMM with a per-row scale in the epilogue. `quantize_vit(extractor)` |
| `decoder_fusion.py` | ConvRefiner decoder fusions for the int8 preset (`fuse_producers`: depthwise + ReLU + static INT8 as `ops.depthwise5x5_relu_int8` for the 256 / 512 / 768-channel blocks; `fuse_refiners`: INT8 upsample-concat into `block1.0`, residual `/1.4` folded into `out_conv`); speed (interleaved) and keypoint / descriptor agreement |
| `vit_fusion_probe.py` | the ViT alone: hand-written kernels vs torch.compile over plain torch ops vs the hybrid (torch-op quantize fused by Inductor + the Triton GEMM) |
| `mega1500_vit_dynamic.py` | MegaDepth-1500 matches for a combination (`--sdpa`, `--matcher-attention`, `--decoder-fusion`) in `run_sparse.py --save-matches` format |
| `paired_auc.py` | paired bootstrap of ΔAUC between two `run_sparse.py --save-errors` files (10 seeds) |
| `speed_vit_dynamic.py` | pair time (2 extracts + match) of bf16 / int8 / int8 + dynamic ViT / + sage-fp8 (ViT, matcher, both) / fp8, interleaved |
| `pipeline_breakdown.py` | per-module CUDA-event time of extract and match |
| `sol_target.py`, `sol_analyze.py` | ncu speed-of-light per kernel: compute- vs memory-limited time per module |

MegaDepth protocol (upstream LoMa's): 1024 px, 4096 keypoints, OpenCV RANSAC 0.5 px, 10 shuffled seeds
(17 29 42 101 211 307 401 503 601 701), paired bootstrap; gate: AUC@5 drop < 0.5 vs bf16. A single run resolves about
±0.5 AUC@5, so judge variants by the paired comparison, not by separate runs.

```sh
python benchmarks/experimental/loma/mega1500_vit_dynamic.py --variant l --matcher-attention sage-fp8 --decoder-fusion \
    --out results/loma/matches/loma-l_X.npz
python benchmarks/accuracy/run_sparse.py --from-matches results/loma/matches/loma-l_X.npz --estimator opencv \
    --ransac-thr 0.5 --seeds 17 29 42 101 211 307 401 503 601 701 --workers 20 \
    --save-errors results/loma/matches/errors_loma-l_X_off0.npy --out results/loma/matches/pose_loma-l_X_off0.json
python benchmarks/experimental/loma/paired_auc.py results/loma/matches/errors_loma-l_bf16_off0.npy \
    results/loma/matches/errors_loma-l_X_off0.npy
```

## Results on sm_120 (RTX 5070 Ti, eager, 2026-10-31)

Pair = 2 extracts + match, one MegaDepth pair, 4096 keypoints, ms (best of interleaved rounds in one process).

| loma-l | pair ms | vs bf16 | ΔAUC@5 vs bf16 [95%] |
|---|---|---|---|
| bf16 | 157.8 | 1.00x | |
| int8 (preset) | 138.4 | 1.14x | -0.42 |
| int8 + dynamic ViT | 109.7 | 1.44x | -0.33 [-0.80, +0.12] |
| + matcher sage-fp8 (`components.matcher.optimize.attention=sage-fp8`) | 101.3 | 1.56x | -0.25 [-0.75, +0.25] |
| + ViT sage-fp8 as well | ~96 (1.66x vs bf16 in its own run) | | -0.37 [-0.84, +0.15]; ViT sage alone -0.55 (near the gate) |
| fp8 (preset) | 109.4 | 1.44x | -1.2 (dynamic-ViT int8 is +0.96 better) |
| + decoder fusion A+B+D | extract -2.8 ms (-5.6 per pair) | | running |

Other variants (int8 + dynamic ViT, pair ms / vs bf16 / ΔAUC@5): b 94.1 / 1.46x / +0.23, g 129.8 / 1.50x / +0.05,
r 94.5 / 1.47x / -0.11.

Findings that may not transfer and are worth re-measuring per architecture:

- **Roofline** (ncu, loma-l): bf16 is 72% compute-limited; int8 + dynamic ViT + sage is 54% memory-limited (glue:
  elementwise, standalone quantize kernels, conv-decoder passes). The 5070 Ti has ~88 TFLOPS BF16 and 896 GB/s; L20
  (sm_89, ~120 TFLOPS BF16 / 864 GB/s) and H20 (sm_90, ~148 TFLOPS BF16 / 4 TB/s) sit at very different ratios, so the
  split between GEMM / attention time and glue time, and which fusions pay, will differ.
- **Dynamic ViT GEMM** uses `backends.triton.gemm._int8_gemm_config` tiles (has sm_89 / sm_90 branches); the row
  quantize is one program per row.
- **ViT glue fusion**: compiling plain torch ops fuses the glue nearly ideally but `torch._int_mm` writes INT32 and
  loses; the hybrid (quantize in torch ops, Triton GEMM with the scales in its epilogue, compiled) was 16.45 -> 15.55
  ms per ViT.
- **Decoder**: the depthwise INT8 producer dispatches to TileLang on sm_120 (needs TMA: sm_90 yes, sm_89 no -> Triton;
  `_depthwise_tile`'s group-major order was tuned on an L20). TileLang JIT-compiles per input shape (first pairs of a
  MegaDepth run are slow). `INT8_CHAINED_CAPABILITIES` (quantized conv writing the next conv's INT8 input from its
  epilogue) is on for sm_89 / sm_90 and off on sm_120, so the decoder's `block1` quantize may already be fused there.
  Compiling the decoders was slower on sm_120 (`_upsample_combine_kernel` 0.80 -> 7.26 ms from the compiled graph).
- **SageAttention**: sage-fp8 (INT8 QK, FP8 PV) needs FP8 (sm_89+); sm_90 has its own sage kernels.
- Compiled LoMa was slower than eager on sm_120, and the quantized extractors do not compile at all (autotuned conv
  kernels with `prune_configs_by` under Dynamo: `ListVariable(...) is already tracked for mutation`).
