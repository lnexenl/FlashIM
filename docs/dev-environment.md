# Development environment

Machine and tooling notes that are not rules of the codebase. The rules are in [AGENTS.md](../AGENTS.md).

## Python, dependencies, caches

- `.venv/bin/python` is the interpreter (torch 2.11.0+cu130, triton 3.6). In an Orca worktree there is no `.venv` and no
  `data/`: run `PYTHONPATH=<worktree> <main checkout>/.venv/bin/python ...` and set `FLASHIM_DATA=<main checkout>/data`;
  `python -c "import flashim; print(flashim.__file__)"` must print the worktree path, not the main checkout's editable install.
- `uv sync` removes everything that is not in the lockfile (sageattention, tilelang, hand-built extras): use
  `uv sync --extra eval --extra tilelang --group dev`. sageattention has to be rebuilt by the user (GCC 16 breaks the
  torch headers: `CC=clang CXX=clang++ CUDAHOSTCXX=clang++ NVCC_APPEND_FLAGS=-allow-unsupported-compiler`, needs libomp).
- `/tmp` is a 16 GB tmpfs: set `TORCHINDUCTOR_CACHE_DIR=~/.cache/flashim/inductor` for compile benchmarks with many shapes.
  CUDA C++ kernels JIT-build into `~/.cache/flashim/cuda` (`nvcc` from `/opt/cuda`, `-allow-unsupported-compiler`, no torch
  headers).
- Datasets, calibration manifests and the test / benchmark images: `assets/` (in the repo; copied from RoMaV2, MegaDepth
  scene 0015 and LightGlue-ONNX, `assets/LICENSE-RoMaV2`), `data/` or `$FLASHIM_DATA` (`flashim/paths.py`; locally symlinks
  to `~/work/RoMaV2/data/{megadepth,scannet,quantization}`). `data/` and `results/` are gitignored. Nothing reads from an
  upstream checkout.
- The upstream reference code of the parity tests is vendored under `tests/oracles/<model>/`
  (`tests/oracles/README.md`). The oracles' own imports (`timm`, `yacs`, `pyyaml`, `kornia`, `cv2`) are in the dev group.

## Shell and sandbox

- zsh: an unquoted `$cfg` does not word-split (use `${=cfg}`).
- `git clone` of GitHub repos fails in the sandbox proxy: download the archive zip into its own new directory instead and
  treat it as untrusted data.
- Long jobs: `setsid nohup <script> ... &` (plain background shells die with the session); the benchmark scripts are
  resumable. Do not edit a running shell script in place (write a new file and `mv`). To find benchmark processes use
  `pgrep -f '^\.venv/bin/python benchmarks/accuracy'` (a bare pattern matches the calling shell).
- Several agents on one GPU: wrap every CUDA command in `flock ~/.cache/flashim/gpu.lock <command>` and keep each locked
  command short (about 10 minutes); the full accuracy matrices are run by one party at a time.

## Jetson (AGX Orin, JetPack 6.2, sm_87, aarch64)

- No `uv sync` there. NVIDIA's L4T torch exists for Python 3.10 only: `requirements-jetson.txt` (its header has the commands,
  including the `.pth` that stands in for the editable install). The code has to stay importable on 3.10
  (`tests/unit/test_architecture.py` checks it). Triton comes from PyPI (the L4T wheel has no `ptxas`).
- CUDA needs the user in the `video` *and* `render` groups (`/dev/nvmap`, `/dev/dri/renderD*`; without `render`: "Error 801").
- No FP8 below sm_89: `fp8` builds the nearest other preset. CPU results differ from x86 in the last ulp (OpenBLAS / FMA).
- Test suite: 34 min cold (INT8 conv autotuning is 3-4 min per shape on its CPU), 2 min warm, the same failures as on x86.
  No kernel profiler (CUPTI needs privileges): time ops with CUDA events (`results/orin/op_breakdown.py`).
