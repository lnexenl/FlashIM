#!/usr/bin/env bash
# FlexGlue MegaDepth-1500 accuracy matrix: the dense match() path (detector-free, KDE-sampled, 5000 matches) at
# FlexGlue's own 512x512 resolution, poselib 1.0px, RANSAC seeds 17 / 29 / 42 on the same matches. Resumable.
#   setsid nohup benchmarks/accuracy/flexglue_matrix.sh > results/accuracy/flexglue_matrix.log 2>&1 &
cd "$(dirname "$0")/../.."
mkdir -p results/accuracy
presets=("$@"); [ ${#presets[@]} -eq 0 ] && presets=(fp32 bf16 fp8 int8)
for preset in "${presets[@]}"; do
  out=results/accuracy/flexglue_mega1500_${preset}.json
  [ -s "$out" ] && { echo "skip $preset"; continue; }
  .venv/bin/python benchmarks/accuracy/run_sparse.py --model flexglue --preset "$preset" --resize 0 --top-k 5000 \
    --seeds 17 29 42 --save-errors "results/accuracy/flexglue_mega1500_${preset}_errors.npy" --out "$out.tmp" \
    > "results/accuracy/flexglue_mega1500_${preset}.log" 2>&1 && mv "$out.tmp" "$out" && echo "done $preset" \
    || echo "FAILED $preset"
done
echo "matrix finished"
