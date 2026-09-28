#!/usr/bin/env bash
# RDD + LightGlue MegaDepth-1500 accuracy matrix (resumable: existing result files are skipped).
#   setsid nohup benchmarks/accuracy/lightglue_matrix.sh [suffix] > results/accuracy/lightglue_matrix.log 2>&1 &
# RDD runs as its default fp8 preset in every row; only LightGlue's preset varies. The pipeline has no sampling
# randomness and OpenCV's RANSAC ignores the seed here, so one run per config.
cd "$(dirname "$0")/../.."
mkdir -p results/accuracy
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/flashim/inductor}
suffix=${1:+_$1}
run() {  # tag, args...
  local tag=$1; shift
  local out=results/accuracy/lightglue_${tag}${suffix}.json
  [ -f "$out" ] && return
  .venv/bin/python benchmarks/accuracy/run_sparse.py --model rdd-lightglue "$@" --out "$out.tmp" && mv "$out.tmp" "$out"
}
run fp32 --preset fp32
run bf16 --preset bf16
run fp8  --preset fp8
run int8 --preset int8
