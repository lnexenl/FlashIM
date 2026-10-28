# NVIDIA architectures, Ampere to Blackwell: what a FlashIM kernel can rely on

Companion to `optimize-kernel`. `kernelwiki` and `ncu-report-skill` cover sm_90 / sm_100 only; this file supplies the
rest so a kernel is designed for the hardware it will run on. Items marked **(verify)** are from memory of NVIDIA /
CUTLASS documentation and were not checked on hardware here. Check them (`nvcc -arch=... `, `ncu --query-metrics`,
`torch.cuda.get_device_properties`) before building a design on them. The only GPU available locally is **sm_120**
(RTX 5070 Ti); everything else can be compile-checked (nvcc 13.4: sm_75 ... sm_121) but not run.

## Feature matrix

FlashIM `Feature` flags come from `flashim/platforms/nvidia.py::isa_features`. Add a new flag only for an instruction
family a kernel here has actually run on.

| Arch (cc) | Parts | MMA path | Async copy | Low precision | FlashIM features today |
|---|---|---|---|---|---|
| Ampere sm_80 | A100, A30 | `mma.sync` m16n8k16 (fp16/bf16), m16n8k8 (tf32), m16n8k32 (int8) | `cp.async` (LDGSTS), no TMA | fp16/bf16/tf32/int8; **no FP8** | `MMA_SYNC` |
| Ampere sm_86 / sm_87 | RTX 30xx, A10/A40, Orin | same as sm_80 | `cp.async` | same; **no FP8** | `MMA_SYNC` |
| Ada sm_89 | RTX 40xx, L4/L40 | same `mma.sync` + FP8 (E4M3/E5M2) `mma.sync` | `cp.async` | FP8 via `mma.sync`, `torch._scaled_mm` (rowwise) | `MMA_SYNC`, `FP8_E4M3`, `FP8_ROWWISE_SCALED_MM` |
| Hopper sm_90a | H100/H200 | `wgmma` (warpgroup, async, smem operands) + legacy `mma.sync` | TMA (+ multicast), mbarrier, clusters/DSMEM | FP8 wgmma | + `TMA`, `WGMMA` |
| Datacenter Blackwell sm_100a / sm_103a / sm_110 | B200, B300 (GB300), Thor | `tcgen05.mma` (accumulator in TMEM, single-thread issue, 2-CTA pairs); legacy `mma.sync` kept | TMA, CLC (cluster launch control) | MXFP8 / MXFP6 / MXFP4 / NVFP4 block-scaled | + `TCGEN05`, `MXFP8_SCALED_MM` |
| Consumer Blackwell sm_120a / sm_121 | RTX 50xx, RTX PRO Blackwell, GB10 (Spark) | `mma.sync` only, incl. block-scaled variants (`kind::mxf8f6f4`, nvf4) **(verify)**; **no wgmma, no tcgen05/TMEM** | TMA + mbarrier | FP8, MXFP8 via cuBLAS `scaled_mm`; FP4 **(verify)** | `MMA_SYNC`, `TMA`, `FP8_*`, `MXFP8_SCALED_MM` (no WGMMA, no TCGEN05) |

sm_120 is **not** "small Blackwell datacenter": kernels written for tcgen05/TMEM (kernelwiki's Blackwell pages,
FlashAttention-4, DeepGEMM sm_100 paths) do not port. Its design space is Ampere/Ada-style `mma.sync` register-fragment
kernels + TMA/mbarrier pipelines.

## Resources per SM (order of magnitude, **(verify)** against `cudaGetDeviceProperties` for the exact part)

| Arch | Max smem/block | Smem+L1 per SM | Regs/SM | Notes |
|---|---|---|---|---|
| sm_80 | ~163 KB (opt-in) | 192 KB | 64K x 32-bit | 40 MB L2 (A100); HBM2e |
| sm_86 / 87 | ~99 KB | 128 KB | 64K | doubled FP32 units; GDDR6/X |
| sm_89 | ~99 KB | 128 KB | 64K | large L2 (up to 72 MB): fits many working sets |
| sm_90 | ~227 KB | 256 KB | 64K | clusters up to 8 (16 opt-in), `setmaxnreg` for warp specialization |
| sm_100 / 103 | ~227 KB | 256 KB + 256 KB TMEM | 64K | clusters, 2-SM MMA |
| sm_120 / 121 | ~99 KB **(verify)** | 128 KB | 64K | cluster size 1 in CUTLASS SM120 kernels **(verify)**; GDDR7 |

Query the real value: `torch.cuda.get_device_properties(0).shared_memory_per_block_optin`. Set
`cudaFuncAttributeMaxDynamicSharedMemorySize` for CUDA C++; Triton `num_stages` x tile bytes must fit it, otherwise the
autotuner silently drops configs (an out-of-resources config is not a slow config).

## Design guidance by tier

**Ampere / Ada (sm_80-89).** Memory-latency hiding is done with `cp.async` multi-stage pipelines (2-4 stages) and
`ldmatrix`; there is no producer/consumer warp specialization with TMA. Triton `num_stages` maps to this. Keep occupancy
in mind: 2-3 CTAs/SM is normal. sm_86/89 have ~100 KB smem, so tiles tuned on A100 (163 KB) often do not fit: parametrize
stages by the measured `shared_memory_per_block_optin`. FP8 only from sm_89 (`Feature.FP8_E4M3`); on sm_80/86 an FP8
storage format must be dequantized to bf16 before `mma.sync` (a memory-bandwidth win only). INT8 `mma` exists from
Turing on and runs at 2x fp16 rate; Triton int8 dot is `DeviceSpec(min_capability=(7, 5))` here.

Run on a Jetson AGX Orin (sm_87, aarch64, JetPack 6.2, Triton 3.6; 2026-10-07): the BF16 / FP32 / INT8 Triton
kernels and the CUDA C++ `raco_resblock` pass their tests against the references there. `tl.dot` lowers to `mma.sync`
m16n8k16 (BF16) and m16n8k32 (INT8). Triton refuses `tl.float8e4nv` outright ("not supported in this architecture"),
so a kernel that only *writes* E4M3 (a quantize producer) needs `requires=[Feature.FP8_E4M3]` as much as an FP8 MMA
does. `fp8` is therefore replaced by the nearest other preset on such GPUs (`build.NEAREST`: `int8`, else `bf16`): on
the torch references RoMaV2 `fp8` took 3334 ms per pair there, INT8 (the `int8-full` preset of the time) 1122 ms, `bf16`
1981 ms.

Launch choices measured there (timing with CUDA events only: CUPTI refuses unprivileged users, so no ncu and no
`torch.profiler` kernel times; `results/orin/op_breakdown.py` times every dispatched op with a synchronize instead):

- INT8 GEMM: the 128 x 128 tile with 4 warps (2-11% over 8 warps, 7-25% over the sm_89 / sm_90 64-row picks).
- INT8 conv chaining pays as on Ada (7% on an eight-conv chain); it is enabled for sm_87.
- The 512-channel depthwise producer is flat between the group-major and group-minor grids (within 3% either way,
  2 x 512 x 200 x 200 and 256 x 256): the Ada pick is not carried over.
- Host overhead is large: a Triton launch costs about 80 us of CPU, so a 300 x 256 INT8 GEMM (0.083 ms) loses to
  cuBLAS BF16 (0.044 ms), and conv autotuning re-timed its candidates in every process (10 s for RoMaV2's INT8 VGG).
  The conv kernels now keep their timings in Triton's cache (`cache_results`, keyed by architecture): first forward
  12.5 -> 2.7 s; `FLASHIM_AUTOTUNE_CACHE=0` re-times.
- `engine=compile` works (Inductor warns "Not enough SMs to use max_autotune_gemm mode": 16 SMs) and buys little once
  the fused kernels are in: RoMaV2 INT8 (the `int8-fast-vgg` preset of the time) 667 -> 655 ms (188 s of
  compilation), SuperPoint `bf16` 64.9 -> 58.9 ms, `int8` 48.5 -> 46.1 ms. `compile-full` was not run there.
- Memory-bound elementwise chains are slow next to fused kernels (LPDDR5 shared with the CPU): DINOv3's upstream
  RoPE math 1.54 ms per call against 0.46 ms for the fused `rope_qk` kernel (RoMaV2's `fused_rope` pass).

Environment: `requirements-jetson.txt`.

Measured on an L20 (sm_89, Triton 3.6, 2026-10-06):

- Triton lowers FP32 -> E4M3 through a *truncated* FP16 below sm_90 (0.4% wrong codes); use `to_e4m3`
  (`backends/triton/quantize.py`), which emits `cvt.rn.satfinite.e4m3x2.f32` by hand.
- One E4M3 `mma.sync` (k = 32) is only accurate to about 1e-4 of sum|products| (FP32 matmul: 1e-8). Compare a new FP8
  kernel bitwise against the existing Triton FP8 path, not against the FP32 reference with a tight tolerance.
- There is no block-scaled MMA: MXFP8 costs one MMA per 32-wide scale block plus an FP32 scale of its result. That
  scaling is about 13% of `_fp8_gemm_kernel`. Write the K loop as a *run-time* loop Triton can pipeline:
  `mxfp8_conv2d` gained 1.2-3.7x from turning its unrolled (tap, block) loops into one such loop; the same change
  lost on INT8 (accumulation fused into the MMA already) and on the fused MLP kernel.
- ncu (`results/profile/2026-10-06_mxfp8_conv2d_sm89/REPORT.md`): after that change `_mxfp8_conv2d_kernel` and
  `_fp8_gemm_kernel` are tensor-pipe bound (62-83% of peak, `math_pipe_throttle` the top stall, `long_scoreboard`
  3-9%); the FP8 ceiling is about 117 TMAC/s. Non-root ncu works on `einstein` and `infinite-frontier` only
  (`grep RmProfilingAdminOnly /proc/driver/nvidia/params`); binary `/usr/local/cuda-13.3/bin/ncu`, Python module in
  `/opt/nvidia/nsight-compute/2026.2.1/extras/python`. Use the autotuned kernel's `.fn` with an explicit tile in the
  driver so `-k regex:<kernel> --launch-skip 3 -c 1` profiles one known config.
- ncu on the remaining kernels (`results/profile/2026-10-06_sm89_kernels/REPORT.md`): several are DRAM-bound
  (650-830 GB/s is the ceiling). What that led to: a two-pass `dual_softmax_mutual` for matrices over 256 MiB
  (1.25x on LoFTR's 1.1 GB matrix; slower below, where the matrix stays in L2), quantized convs writing the next
  conv's input from their epilogue (`feeds` / `quantize_for`, bit-identical, up to 2.6% end to end), a group-major
  grid for 512-channel depthwise producers (1.06-1.17x), and 32-wide chunks for INT8 convs whose channel count pads
  badly (LoFTR 196 channels: 1.45x). A high `no_instruction` share (42% in `_refiner_input_kernel`) is not by
  itself a reason to make loops run-time: there the unrolled taps index registers statically, and the run-time
  version was 0.65-0.85x and not bit-identical.
- GEMM tile sizes tuned elsewhere are 7-27% off here (128 x 64 x 32 FP8, 64 x 64 x 64 INT8), 1.4-1.9x for short K
  loops with 64 x 64 tiles; depthwise / deformable-attention tiles are flat. Timing noise between measurements in one
  process is about 6%: interleave old and new.
- `TRITON_OVERRIDE_ARCH=sm120` plus a faked `torch.cuda.get_device_capability` compiles another arch's code path for
  PTX comparison (compile check only).

**Hopper (sm_90a).** Build with `-arch=sm_90a` (the `a` is required for wgmma / setmaxnreg; PTX for `sm_90` will not
contain them). Use TMA descriptors + mbarrier, warp-specialized producer (`setmaxnreg.dec`) and consumer warpgroups
(`setmaxnreg.inc`), wgmma with accumulators in registers. Triton 3.x emits wgmma/TMA on sm_90 for `tl.dot` (tensor
descriptors via `tl.make_tensor_descriptor`). kernelwiki's Hopper pages apply directly.

Measured on an H20 (sm_90, Triton 3.6, 2026-10-07; raw numbers in `results/slurm/h20_tune/`):

- Triton lowers a `tl.dot` tile with 64 or more rows to wgmma and a smaller one to `mma.sync`. For E4M3 the two round
  differently: `mma.sync` matches the exactly rounded BF16 result (4 of 4.2M outputs of a 256 -> 1024 linear differ),
  wgmma differs in 2.6%. FP8 kernels are therefore bit-identical only among tiles of one row class (they are across
  all tiles on sm_89): compare against the same class, and keep an autotuned FP8 kernel's candidates in one class
  (`_mxfp8_conv2d_tiles`), or its output bits depend on the autotune timing.
- wgmma tiles are 2-2.7x faster than 32-row `mma.sync` tiles on large FP8 GEMMs, so a fused kernel built on 32-row
  tiles loses to a chain of per-op kernels (`mxfp8_gelu_mlp`: 0.563 against 0.308 ms at 30000 rows; sm_90 runs the
  chain, `triton_chain`), and the TileLang `mxfp8_linear` (`mma.sync`) loses to Triton (same bits, 2-30% slower).
- FP8 GEMM: 64 x 128 tiles with one 32-wide K block per iteration and 3 stages, 7-26% over the sm_120 defaults; the Ada
  pick (128 x 64) is slower than those defaults here. INT8 GEMM: the Ada pick (64-row tiles, 4 warps) wins here too
  (9-25%).
- Carried over from Ada after measuring: the fused MXFP8 conv above 36 dot calls (1.4-2.1x over im2col + GEMM) and conv
  chaining (1-6% on a conv pair). Not carried over: the two-pass `dual_softmax_mutual` (1.1-2.4x *slower* at 95 MiB
  to 1 GiB; three passes stay) and the group-major depthwise grid (TileLang wins dispatch through `Feature.TMA` and
  is the fastest: INT8 26-38% over Triton).
- Accuracy of the above (MegaDepth-1500, 3 RANSAC seeds, fp8 presets of 14 sparse models before -> after, eager):
  AUC@5 -0.0009 .. +0.0076, mean +0.0012; every int8 preset and RoMaV2 are bit-identical
  (`results/accuracy/h20_tuning/`, `results/hashes/h20_*`). The fp8 gaps to BF16 of XFeat (-0.014), XFeat+LighterGlue
  (-0.14) and SuperPoint+LightGlue (-0.035) are the same on an L20: not a Hopper effect.
- SageAttention: build with `TORCH_CUDA_ARCH_LIST="8.9+PTX"` to serve sm_89 and sm_90 from one install (the driver
  JITs the PTX on Hopper); `8.9;9.0` does not compile (every gencode is passed to its Hopper-only extension).

**Datacenter Blackwell (sm_100a / 103a / 110).** `-arch=sm_100a` (`sm_103a`, `sm_110a`): tcgen05 is arch-specific and
not forward-compatible between the 10x and 12x families. Accumulators live in TMEM, one thread issues the MMA,
2-CTA pairs share B. NVFP4 / MXFP block-scaled MMAs take scale factors in TMEM. This is where kernelwiki and
ncu-report-skill are written for. Local hardware cannot run it: compile-check only.

**Consumer Blackwell (sm_120a / 121).** `-arch=sm_120a` (`sm_121a` for GB10). Design like Ada with extras: TMA +
mbarrier pipelines (this is why `TILELANG` kernels here require `Feature.TMA`), FP8 and block-scaled MXFP8 through
cuBLAS (`Feature.MXFP8_SCALED_MM`), no TMEM, no wgmma. Measured on this repo's RTX 5070 Ti: see docstrings in
`flashim/backends/{triton,tilelang}/gemm.py`, and the note in `AGENTS.md` that quantized im2col conv is slower than BF16
on the small nets. TileLang gotchas (width <= 16 miscompiles, fence.proxy.async before TMA refill) are in `AGENTS.md`.

## Registering per architecture

- Prefer `requires=[Feature.X]` over `min_capability` whenever the feature is not monotonic. Example: `WGMMA` is
  sm_90 only, `TMA` is sm_90 and sm_120 but not sm_100-with-tcgen05-only designs. A Hopper wgmma kernel must
  `requires=[Feature.WGMMA]`, never `min_capability=(9, 0)`, or it will be selected on sm_100/sm_120 where it does not run.
- One impl per architecture family is normal (`triton` generic + `tilelang_hopper` + ...). Give the specialized impl
  a higher `priority=` and keep the generic one as its fallback; `op.explain(*args)` shows which one is chosen.
- CUDA C++ arch flag from `platform.capability`: `sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100a`, `sm_103a`, `sm_110a`,
  `sm_120a`, `sm_121a`. Include the arch in the JIT cache key (`~/.cache/flashim/cuda`) so a cubin for one arch is never
  loaded on another.
- Triton autotune configs must be filtered by the smem budget above; key tuning caches by arch.
- Compile-check without hardware: `nvcc -arch=sm_90a -cubin -o /dev/null k.cu` (repeat per arch); for Triton,
  `triton.compile(src, target=triton.backends.compiler.GPUTarget("cuda", 90, 32))`. A successful compile proves only
  that the arch accepts the instructions.

## Profiling on non-sm_100 GPUs (overrides for ncu-report-skill)

- `ncu-report-skill/reference/08-b200-metric-names.md` lists sm_100 names. Other archs differ (notably tensor-pipe and
  L1TEX names). Enumerate first: `ncu --query-metrics --chip <ga100|ga102|ad102|gh100|gb100|gb202>` (chip names checked
  against `ncu --list-chips` here), or `action.metric_names()` in the python API. Treat every metric named in the
  skill as a hint, not a fact, until it returns non-`None` on the local GPU.
- The tensor-pipe utilization metric family depends on the MMA type in use (legacy `mma.sync` vs wgmma vs tcgen05): if a
  tensor metric reads 0 while the kernel is a GEMM, look at the other family before concluding the tensor cores are idle.
- Rooflines are per part: read peak bandwidth / FLOPs from `nvidia-smi -q -d CLOCK`, the datasheet and ncu's
  `sm__throughput` / `gpu__compute_memory_throughput` percent-of-peak, not from the B200 numbers in the skill.
- Permissions: ncu needs `NVreg_RestrictProfilingToAdminUsers=0` or root; the skill's `09-common-issues.md` covers this.
- Consumer GPUs boost and throttle: lock clocks only if run-to-run jitter is large (`ncu --clock-control base` is the
  default and is the more reproducible choice).
