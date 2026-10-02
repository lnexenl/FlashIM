---
name: optimize-kernel
description: Use when the user asks to optimize, write, port or speed up a FlashIM kernel or op (Triton / TileLang / CUDA C++ / cuBLAS) on any NVIDIA architecture from Ampere (sm_80) to Blackwell (sm_100/sm_120), or says "kernel design agent" / "KDA". Drives the KDA loop (research, implement, verify, profile, iterate, promote) with FlashIM's op registry, tests, accuracy gate and per-architecture notes, and pulls in kernelwiki and ncu-report-skill.
---

# optimize-kernel — the KDA loop for FlashIM

An adaptation of NVlabs/mit-han-lab *kernel-design-agents*: research, implement, verify, profile, iterate, promote.
Everything is a **candidate** that must pass the gates below before it replaces what is registered.
`AGENTS.md` rules (layering, naming, `Op` registry recipe, environment gotchas) always apply; this skill does not restate them.

## 0. Fill in the task (ask the user only for what you cannot derive)

| Field | Default in this repo |
|---|---|
| Objective | op name (`ops/<category>.py`) + shapes/dtypes from the real model call sites (`op.explain(*args)`, `ops.call_counts()`), never toy shapes |
| Target archs | the local GPU (`current_platform().describe()`), plus any archs the user names. See [`references/nvidia-archs.md`](references/nvidia-archs.md) |
| Constraints | language (see 2), no new hard dependency, torch reference stays the oracle |
| Validation | 3. correctness gate |
| Promotion | 5. promotion gate |

State the filled-in task back in 3-5 lines before writing code.

## 1. Research (before coding)

1. `Read` the op declaration + torch reference, the currently registered impls (`flashim/backends/*/<category>.py`) and their tests.
2. Identify the target arch(es) and read the matching section of [`references/nvidia-archs.md`](references/nvidia-archs.md). **Design for the features the arch has** (`Feature.*`), not for the newest arch.
3. Query **kernelwiki** for prior art (`python3 scripts/query.py --architecture sm90 "<topic>"`, run from `.agents/skills/kernelwiki`). Its corpus is Hopper/Blackwell-datacenter only; for Ampere / Ada / sm_120 it gives the algorithmic pattern (pipelining, tiling, fusion), not the instructions. Say so in the notes rather than copying tcgen05/wgmma ideas onto hardware without them.
4. If a baseline is slow, **profile before hypothesizing** (step 4) instead of reading more wikis.

## 2. Pick the kernel language (in this order unless there is a reason)

1. **Triton**: visible to Inductor, cheapest to iterate, portable across sm_80..sm_120. Default for elementwise / reduction / quantize / small GEMM / attention variants.
2. **TileLang**: TMA / mbarrier pipelines (requires `Feature.TMA`, so sm_90+ and consumer Blackwell). Follow the TileLang gotchas in `AGENTS.md`. Wrap in `torch.library.custom_op` with a fake.
3. **CUDA C++** (JIT via nvcc, `csrc/`): only when PTX-level control is needed (`mma.sync` fragments, `cp.async`, `ldmatrix`, inline PTX). Wrap in `custom_op`. Set `-arch` per arch from the table in the arch reference (`sm_90a`, `sm_100a`, `sm_120a` need the `a`).
4. **Vendor libs** (cuBLAS / `scaled_mm` via `backends/torch_native/`) when they are already near roofline: measure before writing anything.

Register with `requires=[Feature...]` (not capability ranges when the feature is not monotonic), `device=DeviceSpec("cuda", ...)`, `when=` for shape eligibility. Keep the torch fallback reachable on every arch you cannot test.

## 3. Correctness gate (a candidate that fails this is discarded, not tuned)

- `.venv/bin/python -m pytest -q -p no:cacheprovider tests/ops -k <op>`: new/updated test of the impl vs `op.get("reference")` over a shape sweep that includes non-multiples of the tile sizes, tiny and odd sizes, and the real call-site shapes. Follow the style of `tests/ops/test_ops_cuda.py` (`_both`, bit-mismatch fractions for quantized outputs, tolerances stated with the reason).
- `op.explain(*args)` shows the new impl is selected for the real shapes, and `ops.call_counts()` shows it runs in the model path under test.
- Lint only the files you touched: `.venv/bin/ruff check <files>`; `--fix` only on those files.
- Refactor-style changes must be bit-identical (see "Verifying changes" in `AGENTS.md`).
- **Other architectures**: only this machine's GPU can be *run*. For each other target arch, at least compile-check (`nvcc -arch=<sm> -cubin`, or Triton `triton.compile` with `GPUTarget("cuda", <cc>)`; nvcc 13.x here covers sm_75..sm_121) and say plainly in the result that it is **compiled, not validated**. Do not add a `Feature` to a platform detector for hardware nobody ran on.

## 4. Profile (evidence for every claim)

- Wall time: `torch.cuda.Event` / `triton.testing.do_bench` with warmup, best-of-N, same inputs as the baseline, interleaved with the baseline. Long GPU runs need the user's OK; compiled RoMaV2 varies 15-25% between processes.
- Bottleneck: **ncu-report-skill** (`.agents/skills/ncu-report-skill`), with these overrides:
  - Put runs in `results/profile/<run_name>/` (gitignored), not `profile/`.
  - Its metric-name reference is sm_100-only. On other archs enumerate names (`ncu --query-metrics`, `action.metric_names()`); see the profiling section of [`references/nvidia-archs.md`](references/nvidia-archs.md).
  - Kernels launched from Triton/TileLang/torch: use a standalone harness or `ncu -k regex:<name> -c 1 --set full` on a small driver script under the scratchpad, `-lineinfo` where you control nvcc flags.
- Name two or three metric values that support the diagnosis, then change one thing at a time.

## 5. Promotion gate

Log every candidate in `results/kernel_runs/<op>/<date>_<slug>/candidates.jsonl`, one JSON object per line:
`{"id", "change", "arch", "correct": bool, "max_err", "mismatch_frac", "ms", "baseline_ms", "speedup", "profile": "<path or null>", "verdict": "kept|discarded", "why"}`.

Promote (make it the default registration / raise `priority=`) only when all hold:
1. Correctness gate passed on the local GPU; other archs compile-checked and labelled as such.
2. Faster than the current best on the **real call-site shapes** by more than run-to-run noise (interleaved best-of-N, two rounds).
3. Accuracy: if numerics differ from the previous impl, report the task metric per `AGENTS.md` (AUC@5 drop < 0.005 vs the BF16 compile baseline, 3 seeds) and get the user's OK before running the long benchmark.
4. Design choices with a *measured* reason go in the docstring (e.g. "BLOCK_N=64 measured 1.3x faster than 128 on sm_120"), not guesses.

Discarded candidates stay in `candidates.jsonl` with the reason so they are not retried.

## 6. Hand-off

Report: what changed, the table of candidates (kept and discarded), numbers with shapes and arch, what was compile-checked vs run, and anything left. Do not commit unless asked. Never touch `benchmarks/accuracy/run.py`.
