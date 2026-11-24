#!/usr/bin/env bash
# MegaDepth-1500 + ScanNet-1500 accuracy gate matrix for integration/perf-all vs dev (f9ce273).
# Resumable (skips any job whose .json and .npy already exist). Acquires ~/.cache/flashim/gpu.lock around each
# individual GPU run (never around the whole matrix).
#
# Full run (launch detached):
#   setsid nohup benchmarks/accuracy/perf_matrix.sh > results/accuracy/perf/matrix.log 2>&1 &
#
# Smoke test (20 pairs/dataset, subset of models):
#   OUT_DIR=results/accuracy/perf_smoke MAX_PAIRS=20 benchmarks/accuracy/perf_matrix.sh orb superpoint
#
# Optional filters (model names):
#   benchmarks/accuracy/perf_matrix.sh romav2 loftr superpoint
set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

PY=${PY:-/home/lnex/work/FlashIM/.venv/bin/python}
OUT_DIR=${OUT_DIR:-results/accuracy/perf}
SEEDS=${SEEDS:-"17 29 42"}
MAX_PAIRS=${MAX_PAIRS:-0}
DATASETS=${DATASETS:-"mega1500 scannet1500"}
GPU_LOCK=${GPU_LOCK:-$HOME/.cache/flashim/gpu.lock}
DEV_COMMIT=${DEV_COMMIT:-f9ce273}
DEV_WORKTREE=${DEV_WORKTREE:-/tmp/flashim-dev-${DEV_COMMIT}}

export FLASHIM_DATA=${FLASHIM_DATA:-$HOME/work/FlashIM/data}
export FLASHIM_IMAGE_DECODER=${FLASHIM_IMAGE_DECODER:-pil}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/flashim/inductor}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT_DIR" "$(dirname "$GPU_LOCK")" "$TORCHINDUCTOR_CACHE_DIR"

# Prepare a clean detached worktree at dev (f9ce273) for baseline runs without touching any existing worktree.
if [ ! -f "$DEV_WORKTREE/flashim/__init__.py" ]; then
  echo "Creating detached dev baseline worktree at $DEV_WORKTREE ($DEV_COMMIT)..."
  git worktree add --detach "$DEV_WORKTREE" "$DEV_COMMIT" || exit 1
fi

# shellcheck disable=SC2206
ds_arr=($DATASETS)
list_args=(--list-jobs --datasets "${ds_arr[@]}")
if [ $# -gt 0 ]; then
  list_args+=(--models "$@")
fi
job_list=$(PYTHONPATH="$REPO_ROOT" "$PY" benchmarks/accuracy/perf_protocol.py "${list_args[@]}") || exit 1
mapfile -t JOBS <<< "$job_list"

echo "Scheduled $(printf '%s\n' "${JOBS[@]}" | grep -c '^RUN|') job specs -> $OUT_DIR (dev=$DEV_COMMIT, new=$(git rev-parse --short HEAD))"

# shellcheck disable=SC2206
seed_arr=($SEEDS)
failed=0

for line in "${JOBS[@]}"; do
  [ -z "$line" ] && continue
  IFS='|' read -r action tree runner dataset model preset resize topk overrides note <<< "$line"
  if [ "$action" != "RUN" ]; then
    echo "$line"
    continue
  fi

  py_path="$REPO_ROOT"
  [ "$tree" = "dev" ] && py_path="$DEV_WORKTREE"

  if [ "$runner" = "dense" ]; then
    for seed in "${seed_arr[@]}"; do
      stem="${tree}_${dataset}_${model}_${preset}_compile_seed${seed}"
      out_json="$OUT_DIR/${stem}.json"
      out_err="$OUT_DIR/${stem}.npy"
      out_log="$OUT_DIR/${stem}.log"
      if [ -f "$out_json" ] && [ -f "$out_err" ]; then
        continue
      fi
      echo "== [$tree] $dataset $model $preset compile seed=$seed${note:+ [$note]}"
      cmd=(
        env "PYTHONPATH=$py_path"
        "$PY" benchmarks/accuracy/run.py
        --dataset "$dataset"
        --model "$model"
        --preset "$preset"
        --engine compile
        --seed "$seed"
        --max-pairs "$MAX_PAIRS"
        --save-errors "$out_err.tmp.npy"
        --out "$out_json.tmp"
      )
      if flock "$GPU_LOCK" "${cmd[@]}" > "$out_log" 2>&1; then
        mv "$out_json.tmp" "$out_json"
        mv "$out_err.tmp.npy" "$out_err"
      else
        rm -f "$out_json.tmp" "$out_err.tmp.npy"
        echo "FAILED [$tree] $dataset $model $preset seed=$seed (see $out_log)"
        failed=$((failed + 1))
      fi
    done
  else
    stem="${tree}_${dataset}_${model}_${preset}"
    out_json="$OUT_DIR/${stem}.json"
    out_err="$OUT_DIR/${stem}.npy"
    out_log="$OUT_DIR/${stem}.log"
    if [ -f "$out_json" ] && [ -f "$out_err" ]; then
      continue
    fi
    echo "== [$tree] $dataset $model $preset (resize=$resize top_k=$topk${overrides:+ overrides=$overrides})${note:+ [$note]}"
    cmd=(
      env "PYTHONPATH=$py_path"
      "$PY" benchmarks/accuracy/run_sparse.py
      --dataset "$dataset"
      --model "$model"
      --preset "$preset"
      --engine eager
      --resize "$resize"
      --top-k "$topk"
      --estimator poselib
      --ransac-thr 1.0
      --max-pairs "$MAX_PAIRS"
      --seeds "${seed_arr[@]}"
      --save-errors "$out_err.tmp.npy"
      --out "$out_json.tmp"
    )
    if [ -n "$overrides" ]; then
      # shellcheck disable=SC2206
      ov_arr=($overrides)
      cmd+=(--overrides "${ov_arr[@]}")
    fi
    if flock "$GPU_LOCK" "${cmd[@]}" > "$out_log" 2>&1; then
      mv "$out_json.tmp" "$out_json"
      mv "$out_err.tmp.npy" "$out_err"
    else
      rm -f "$out_json.tmp" "$out_err.tmp.npy"
      echo "FAILED [$tree] $dataset $model $preset (see $out_log)"
      failed=$((failed + 1))
    fi
  fi
done

if [ "$failed" -gt 0 ]; then
  echo "Completed with $failed failure(s) -> $OUT_DIR"
else
  echo "All scheduled jobs completed cleanly -> $OUT_DIR"
fi
touch "$OUT_DIR/DONE"
