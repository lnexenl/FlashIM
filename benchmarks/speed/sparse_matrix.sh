#!/usr/bin/env bash
# Latency of the sparse / semi-dense models in fp32 / bf16 / int8 / fp8 on one fixed MegaDepth pair
# (benchmarks/speed/extractor.py; every configuration in its own process; resumable: JSON lines are appended once per tag).
#
#   benchmarks/speed/sparse_matrix.sh [model-filter ...]      # ~20 s per configuration, GPU otherwise idle
#
# Unified protocol, same as benchmarks/accuracy/sparse_matrix.sh: 1200 px long side, 4096 keypoints (loftr: all coarse
# matches). The *-lightglue pipelines run extractor and matcher in the same precision; the pair forward time is
# 2 x extract + match (loftr: match). Results: results/speed/unified/sparse_speed.jsonl (results/speed/sparse_speed.jsonl
# holds the older per-model resolutions).
cd "$(dirname "$0")/../.."
OUT_DIR=${OUT_DIR:-results/speed/unified}
mkdir -p "$OUT_DIR"
out=$OUT_DIR/sparse_speed.jsonl
touch "$out"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/flashim/inductor}
filter=("$@")

wanted() {
  [ ${#filter[@]} -eq 0 ] && return 0
  for f in "${filter[@]}"; do [ "$f" = "$1" ] && return 0; done
  return 1
}

run() {  # tag model preset top_k [extra extractor.py args...]
  local tag=$1 model=$2 preset=$3 topk=$4; shift 4
  wanted "$model" || return 0
  grep -q "\"tag\": \"$tag\"" "$out" && return 0
  echo "== $tag"
  .venv/bin/python benchmarks/speed/extractor.py --model "$model" --preset "$preset" --resize 1200 --top-k "$topk" \
    --tag "$tag" "$@" 2> "$OUT_DIR/sparse_${tag}.err" | tail -1 >> "$out" \
    || echo "FAILED $tag"
}

for p in fp32 bf16 int8 fp8; do
  for m in superpoint disk alike xfeat dedode dedode-g dad ripe; do run "$m-$p" $m $p 4096; done
  for m in superpoint disk xfeat; do
    run "$m-lightglue-$p" $m-lightglue $p 4096 --overrides components.extractor.preset=$p
  done
  run "loftr-$p" loftr $p 0
  for v in b128 b l g r; do run "loma-$v-$p" loma-$v $p 4096; done
done
run sift-fp32 sift fp32 4096
run orb-fp32  orb  fp32 4096
for att in sage sage-fp8; do
  run "xfeat-lightglue-bf16-$att" xfeat-lightglue bf16 4096 --overrides components.extractor.preset=bf16 components.matcher.optimize.attention=$att
done
