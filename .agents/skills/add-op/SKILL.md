---
name: add-op
description: Use when adding or changing a FlashIM op, a backend implementation (Triton, TileLang, CUDA C++, cuBLAS), a new backend package or a platform detector. Covers declaration + torch reference, registration, torch.compile visibility, naming, tests and kernel-language gotchas.
---

# add-op

For making an existing kernel faster use `optimize-kernel` (the KDA loop); this skill is the plumbing.

## Add an op

1. Declaration + torch reference in `flashim/ops/<category>.py` (`Op(name, reference[, weights=n, composite=])` plus a
   docstring), exported in `flashim/ops/__init__.py`. The reference is the correctness oracle and the universal fallback.
2. Implementation + registration in `flashim/backends/<backend>/<category>.py`, next to its kernels, with `device=`,
   `requires=` (ISA features, not capability ranges when the feature is not monotonic in the capability), `when=` (per-call
   eligibility), and for ops with bound weights (`Op(weights=n)`) `prepare=` / `accepts=` (call `op.bind(...).warmup()`
   at load time).
3. Test impl vs reference in `tests/ops/` over odd and non-tile-multiple shapes plus the real call-site shapes; check
   `op.explain(*args)` picks the impl and `ops.call_counts()` shows it runs in the model.

Naming: files in `ops/`, `backends/*/` and `quant/` are named by *what they compute* (`attention`, `conv`, `correlation`,
`gemm`, `quantize`), never by the model part using them. An op only one model can use goes in `ops/<model>.py` with a
model-prefixed name (`romav2_refiner_input`, `raco_resblock`). Quant passes are named by the structural pattern
(`transformer_block`, `separable_conv_block`). Two models computing the same thing share one op.

## torch.compile and heavy JIT

- Kernels opaque to `torch.compile` (CUDA C++ via ctypes, TileLang, vendor libs) are wrapped in `torch.library.custom_op`
  with a fake. Triton stays visible to Inductor.
- Heavy JIT frameworks keep their builders in a private `_*_kernels.py` imported on first call (backend loading must stay
  cheap).

## Add a backend or a platform

- Backend: a new package under `flashim/backends/`, a `Backend(...)` entry in `backends/_loader.py` (with the Python
  packages it needs); impls register for the platform's dispatch key and features. Out-of-tree backends use the
  `flashim.backends` entry point.
- Platform: a detector in `flashim/platforms/` (or `register_detector`, entry point `flashim.platforms`) setting a distinct
  dispatch key (`cuda` is NVIDIA only, ROCm is `rocm`) and only features that some kernel has been validated with on that
  vendor.

## Kernel-language gotchas

- TileLang 0.1.14: no `from __future__ import annotations` in TileLang files (a test checks it); width <= 16 miscompiles;
  `T.no_set_max_nreg()`; an explicit `fence.proxy.async` is needed before TMA refills a stage read by plain loads.
- The autotuned conv kernels (`backends/triton/conv.py`) keep their timings in Triton's cache, so a pick made on a busy GPU
  sticks until the kernel source changes: `FLASHIM_AUTOTUNE_CACHE=0` re-times in every process (do that when comparing
  tiles). Editing that file shifts line numbers, which recompiles every kernel in it.
- torch 2.11: `engine="compile"` fails on a graph that calls one `@triton.autotune(prune_configs_by=...)` kernel twice
  (Dynamo `ListVariable` assertion in `wrap_user_defined_obj`); quantized models with autotuned INT8 convs run eager.
- Only the local GPU can be run (sm_120). Other architectures are compile-checked (`nvcc -arch=<sm> -cubin`, or Triton
  `GPUTarget`) and must be reported as compiled, not validated; never add a `Feature` to a platform detector for hardware
  nobody ran on.
