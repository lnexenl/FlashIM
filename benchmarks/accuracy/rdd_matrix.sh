#!/usr/bin/env bash
# RDD MegaDepth-1500 accuracy matrix (resumable: existing result files are skipped).
#   setsid nohup benchmarks/accuracy/rdd_matrix.sh [suffix] > results/accuracy/rdd_matrix.log 2>&1 &
# The pipeline has no sampling randomness and OpenCV's RANSAC ignores the seed here, so one run per config.
# Compiled runs build one graph per image resolution (24 at 1600px): the Inductor cache goes to disk, not /tmp
# (tmpfs), and uses the default compile mode (~25 s per resolution instead of minutes).
cd "$(dirname "$0")/../.."
mkdir -p results/accuracy
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/flashim/inductor}
suffix=${1:+_$1}
run() {  # tag, args...
  local tag=$1; shift
  local out=results/accuracy/rdd_${tag}${suffix}.json
  [ -f "$out" ] && return
  .venv/bin/python benchmarks/accuracy/run_sparse.py "$@" --out "$out.tmp" && mv "$out.tmp" "$out"
}
fast='{"engine.mode": null}'
run fp32              --preset fp32
run bf16              --preset bf16
run fp8               --preset fp8
run bf16_compile      --preset bf16 --engine compile --extra "$fast"
run fp8_compile       --preset fp8 --engine compile --extra "$fast"
