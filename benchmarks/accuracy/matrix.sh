#!/usr/bin/env bash
# Run the 3-seed x 2-dataset accuracy matrix for the given presets, skipping finished runs.
#   setsid nohup benchmarks/accuracy/matrix.sh bf16 fp8 > results/accuracy/matrix.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
mkdir -p results/accuracy
engine="${ENGINE:-compile}"
for ds in mega1500 scannet1500; do
  for seed in 17 29 42; do
    for preset in "$@"; do
      tag="${ds}_${preset}_${engine}${ATTENTION:+_$ATTENTION}${TAG:+_$TAG}_seed${seed}"
      out="results/accuracy/${tag}.json"
      if [ -s "$out" ]; then echo "skip $tag"; continue; fi
      if .venv/bin/python benchmarks/accuracy/run.py --dataset "$ds" --preset "$preset" --engine "$engine" \
          --seed "$seed" ${ATTENTION:+--attention "$ATTENTION"} --out "$out" > "results/accuracy/${tag}.log" 2>&1; then
        echo "done $tag auc5=$(python3 -c "import json;print(round(json.load(open('$out'))['auc_5'],6))")"
      else
        echo "FAILED $tag"
      fi
    done
  done
done
echo "matrix finished"
