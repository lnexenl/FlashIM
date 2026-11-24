---
name: verify-change
description: Use before declaring a FlashIM change done, when reviewing one, or when setting up a numerics, speed or accuracy comparison. Covers bit-identity checks of refactors, the AUC accuracy gate, the speed protocol, profiling and long GPU jobs.
---

# verify-change

Say in the commit / PR what was verified and what was not (other GPU architectures, `engine="compile"`, datasets).

## Refactors must be bit-identical

`benchmarks/profile/output_hashes.py` fingerprints every model / preset (sha256 of the output tensors and
`ops.call_counts()`). Run it once per tree, in a `git worktree` of the base commit and in the working tree, then compare:

```bash
PYTHONPATH=/path/to/base-worktree python benchmarks/profile/output_hashes.py --out results/hashes/base
python benchmarks/profile/output_hashes.py --out results/hashes/new
python benchmarks/profile/output_hashes.py --compare results/hashes/base results/hashes/new
```

- Run the base tree twice: configurations that differ between two base runs (Triton autotune; use
  `FLASHIM_AUTOTUNE_CACHE=0`) are "not reproducible", not passes.
- Check that the path under test ran: the same `ops.call_counts()`, and no unexpected `model.build_report.fallbacks`
  (a component without the pipeline's preset builds another one).
- The set covers the older models, LoMa and the 2026-11 additions (`--list` prints the names, `--configs <fnmatch>` picks some);
  the file records `fallbacks` and `--compare` prints them next to the call counts. A model that is not in the list: add it.
- In a worktree, check the tree under test is the one imported: `PYTHONPATH=<tree> python -c "import flashim; print(flashim.__file__)"`.
  Keep the base worktree and `results/hashes/` outside `/tmp` (a 16 GB tmpfs).

## Accuracy changes: task-level AUC, not tensor allclose

- MegaDepth-1500 and ScanNet-1500, seeds 17 / 29 / 42, AUC@5 drop < 0.005 vs the BF16 baseline of the same model; HardMatch
  mAA@10px for hard pairs. A 1500-pair AUC@5 delta resolves about +-0.005: decide with a paired bootstrap over pairs, not
  the point estimate (`PASS` / `FAIL` / `BORDERLINE`).
- Models added in 2026-11: `benchmarks/accuracy/b_models_matrix.sh` (resumable, one GPU lock per run) and
  `b_models_report.py`; new models go into `b_models_protocol.py`. `run_sparse.py` is the generic runner (any model that
  returns `Matches`; `--dataset scannet1500`), `run.py` is RoMaV2's.
- Compare numerics in eager in one process: `compile-full` is not reproducible across processes (max-autotune) and two
  compiled instances in one process share Dynamo graphs.
- Long runs: `setsid nohup <script> &`, only with the user's OK; see `docs/dev-environment.md`.

## Speed

- Compiled RoMaV2 varies 15-25% between processes. Compare best-of-N *interleaved* runs of the old and new trees (small N,
  about 2 rounds), same inputs, CUDA events with warmup (`triton.testing.do_bench` for kernels).
- Profile before explaining a slowdown: `torch.profiler` CUDA activities, `ncu` on the suspect kernel and read the limiter
  (e.g. L1/TEX % for gathers). A time table is `benchmarks/speed/speed_table.sh`.
- A speedup claim names GPU / arch, model, preset, engine and shape.

## Tests and lint

- `.venv/bin/python -m pytest -q -p no:cacheprovider tests` (`-m "not cuda"` without a GPU). The vendored LightGlue oracles
  turn off flash SDPA and set `cudnn.deterministic` for the whole process; `tests/conftest.py` restores both after every test.
- `.venv/bin/ruff check <files you changed>`; `--fix` only on files you changed (a directory `--fix` once rewrote the user's
  uncommitted files); preview with `--diff`.
