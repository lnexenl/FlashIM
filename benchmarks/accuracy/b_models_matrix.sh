#!/usr/bin/env bash
# MegaDepth-1500 + ScanNet-1500 AUC gate matrix for the 10 new B-models (eager, poselib 1.0 px, seeds 17 29 42).
# Resumable (skips any job whose .json and .npy already exist). Acquires ~/.cache/flashim/gpu.lock around each
# individual run_sparse.py invocation (never around the whole script).
#
# Full run (launch detached):
#   setsid nohup benchmarks/accuracy/b_models_matrix.sh > results/accuracy/b_models/matrix.log 2>&1 &
#
# Smoke test (30 pairs/dataset):
#   OUT_DIR=results/accuracy/b_models_smoke MAX_PAIRS=30 benchmarks/accuracy/b_models_matrix.sh
#
# Optional filters (by family key or registry model name):
#   benchmarks/accuracy/b_models_matrix.sh roma mast3r omniglue
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
OUT_DIR=${OUT_DIR:-results/accuracy/b_models}
SEEDS=${SEEDS:-"17 29 42"}
MAX_PAIRS=${MAX_PAIRS:-0}
DATASETS=${DATASETS:-"mega1500 scannet1500"}
PRESETS=${PRESETS:-"fp32 bf16 int8 fp8"}
GPU_LOCK=${GPU_LOCK:-$HOME/.cache/flashim/gpu.lock}

export FLASHIM_DATA=${FLASHIM_DATA:-$PWD/data}
export PYTHONPATH=${PYTHONPATH:-$PWD}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/flashim/inductor}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT_DIR" "$(dirname "$GPU_LOCK")"

# shellcheck disable=SC2206
ds_arr=($DATASETS)
# shellcheck disable=SC2206
pr_arr=($PRESETS)
list_args=(--list-jobs --datasets "${ds_arr[@]}" --presets "${pr_arr[@]}")
if [ $# -gt 0 ]; then
  list_args+=(--models "$@")
fi
job_list=$("$PY" benchmarks/accuracy/b_models_protocol.py "${list_args[@]}") || exit 1
mapfile -t JOBS <<< "$job_list"

failed=0
for line in "${JOBS[@]}"; do
  [ -z "$line" ] && continue
  IFS='|' read -r action dataset model preset resize topk overrides fb_note <<< "$line"
  if [ "$action" != "RUN" ]; then
    echo "$line"
    continue
  fi
  stem="${dataset}_${model}_${preset}"
  out_json="$OUT_DIR/${stem}.json"
  out_err="$OUT_DIR/${stem}.npy"
  out_log="$OUT_DIR/${stem}.log"
  if [ -f "$out_json" ] && [ -f "$out_err" ]; then
    continue
  fi

  note=""
  [ -n "$fb_note" ] && note=" [$fb_note]"
  echo "== $dataset $model $preset (resize=$resize top_k=$topk${overrides:+ overrides=$overrides})$note"

  # Build argv cleanly (avoid word-splitting bugs on empty overrides).
  cmd=(
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
    --save-errors "$out_err.tmp.npy"
    --out "$out_json.tmp"
  )
  # shellcheck disable=SC2206
  seed_arr=($SEEDS)
  cmd+=(--seeds "${seed_arr[@]}")
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
    echo "FAILED $dataset $model $preset (see $out_log)"
    failed=$((failed + 1))
  fi
done

if [ "$failed" -gt 0 ]; then
  echo "Completed with $failed failure(s)."
  exit 1
fi
echo "All scheduled jobs completed cleanly -> $OUT_DIR"
