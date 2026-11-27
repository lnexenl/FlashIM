# Compile checks

`compile_cuda.py` uses the flags from the runtime loader and builds one shared library per CUDA source. The three
toolkit jobs install NVIDIA's nvcc and cudart development packages in Ubuntu 24.04. Their toolkit downloads were
63–100 MB and took 12–61 seconds locally. This keeps the devel image's math-library layers out of runner setup.
Both nvcc and TileLang track the newest 13.x toolkit for early warning of compiler regressions; 12.6 and 12.8
exercise fixed minor-version floors. TileLang uses the toolkit package's `/usr/local/cuda-13` symlink.

`compile_triton.py` compiles every eligible specialization in `triton.jsonl` to a cubin. Each row contains the kernel,
signature, constants, argument attributes, compiler options and observed op implementations. The capture evaluates
each implementation's predicates on the original call operands for simulated CUDA platforms with Triton and
the architecture's torch math features. CUDA C++ and TileLang have separate jobs. Replay checks those platforms
against the installed op registry. Direct test launches use their registered calling implementations.
The `sm_110` target is included when Triton's own assembler recognizes it. This job uses the Triton requirement
published for the corresponding CUDA torch wheel, since CPU wheels omit Triton. Each torch release × SM gets
its own runner. `--workers 4` distributes its eligible rows across four compiler processes; local runs default
to one worker and release compiler memory between targets.

Actions caches `~/.triton/cache` per installed torch and Triton version, SM, and hash of the backend sources
and manifest. After an edit it can restore the last cache for the same versions and SM. On both exact and
prefix hits the script still calls `triton.compile` and requires a nonempty cubin for every eligible row.
Triton's internal key includes the kernel and its JIT dependencies, signature, constants, argument attributes,
compiler options, target SM, ptxas version and Triton compiler build. A matching cubin is evidence of a prior
successful compile for that input. A missing key compiles from source. Workers resolve JIT dependency hashes
in kernel-name order so that scheduling cannot change Triton 3.6's keys for kernels that share helpers.

The checked-in launches come from an RTX 5070 Ti (`sm_120`) running PyTorch 2.11.0+cu130 and Triton 3.6.0.
PyTorch 2.10.0 is the lowest tested release whose CPU tests and Triton replay both pass. PyTorch 2.9.0 and
2.9.1 pass the CPU tests, but their Triton 3.5.0 and 3.5.1 compilers reject the registered `linear_fp32`
split-weight specialization on `sm_100` with an accumulator dominance error in `tl.dot`.

To refresh the manifest, set `FLASHIM_RECORD_TRITON` to a fresh JSONL path and run the local GPU tests under the shared
lock, in batches. On the torch 2.11 reference system, `CUDA_PATH` points at a toolkit containing CUDA 12.8's
`nvvm/libdevice/libdevice.10.bc`, which enables the exact GELU tests. Continue with `tests/quant`, `tests/unit`, the
top-level parity tests and each parity model. `merge_triton.py` merges identical compiler inputs across processes.

```bash
mkdir -p results
FLASHIM_RECORD_TRITON="$PWD/results/ops.jsonl" flock ~/.cache/flashim/gpu.lock \
    .venv/bin/python -m pytest -q -p no:cacheprovider tests/ops
.venv/bin/python ci/merge_triton.py results/ops.jsonl results/quant.jsonl results/unit.jsonl results/parity*.jsonl
CUDA_VISIBLE_DEVICES='' .venv/bin/python ci/compile_triton.py
```

The op eligibility probes run with the original tensor arguments during recording. Re-record after changing a
shape-dependent dispatch predicate. Replay checks the installed software's architecture and feature requirements
again. It uses the target Triton's bundled libdevice for compilation. GPU tests check numerical agreement.

`compile_tilelang.py` builds the two depthwise kernels for Hopper and Blackwell and the MXFP8 GEMM for Blackwell.
It supplies an explicit architecture and nvcc compiles device code without loading or launching a CUDA module.
