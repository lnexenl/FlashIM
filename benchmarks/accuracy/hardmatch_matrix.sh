#!/usr/bin/env bash
# HardMatch test split, 3 seeds per configuration, skipping finished runs.
#   setsid nohup benchmarks/accuracy/hardmatch_matrix.sh > results/accuracy/hardmatch_matrix.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
mkdir -p results/accuracy
for cfg in "bf16 compile" "fp8 compile-full"; do
  set -- $cfg
  for seed in 17 29 42; do
    tag="hardmatch_${1}_${2}_seed${seed}"
    out="results/accuracy/${tag}.json"
    if [ -s "$out" ]; then echo "skip $tag"; continue; fi
    if .venv/bin/python benchmarks/accuracy/run_hardmatch.py --preset "$1" --engine "$2" --seed "$seed" \
        --out "$out" > "results/accuracy/${tag}.log" 2>&1; then
      echo "done $tag mAA10=$(python3 -c "import json;print(round(json.load(open('$out'))['mAA_10px'],4))")"
    else
      echo "FAILED $tag"
    fi
  done
done
echo "matrix finished"
