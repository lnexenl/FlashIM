#!/usr/bin/env bash
# benchmarks/profile/sparse_sweep.py over every sparse / semi-dense model in bf16 / int8 / fp8 (one process each,
# GPU otherwise idle). Results: results/profile/unified_sweep/<tag>.json and sweep.log.
cd "$(dirname "$0")/../.."
OUT=${OUT:-results/profile/unified_sweep}
mkdir -p "$OUT"
for p in bf16 int8 fp8; do
  for m in xfeat alike superpoint disk ripe dedode dedode-g dad; do
    .venv/bin/python benchmarks/profile/sparse_sweep.py --model $m --preset $p --out "$OUT" 2>/dev/null | tail -2
  done
  for m in xfeat superpoint disk; do
    .venv/bin/python benchmarks/profile/sparse_sweep.py --model $m-lightglue --preset $p \
      --overrides components.extractor.preset=$p --out "$OUT" 2>/dev/null | tail -2
  done
  .venv/bin/python benchmarks/profile/sparse_sweep.py --model loftr --preset $p --top-k 0 --out "$OUT" 2>/dev/null | tail -1
done
