#!/usr/bin/env bash
# Peak GPU memory of every sparse / semi-dense config of sparse_matrix.sh (one process each), plus the XFeat+LightGlue
# Sage attention variants. Appends to results/accuracy/unified/vram.jsonl; resumable (finished configs are skipped).
#
#   setsid nohup benchmarks/accuracy/vram_matrix.sh > results/accuracy/unified/vram_matrix.log 2>&1 &
cd "$(dirname "$0")/../.."
OUT=${OUT:-results/accuracy/unified/vram.jsonl}
touch "$OUT"

run() {  # label model preset top_k [extra vram.py args...]
  local label=$1 model=$2 preset=$3 topk=$4; shift 4
  grep -q "\"label\": \"$label\"" "$OUT" && return 0
  echo "== $label"
  .venv/bin/python benchmarks/accuracy/vram.py --model "$model" --preset "$preset" --top-k "$topk" --label "$label" "$@" --out "$OUT" 2>&1 | grep -v -i warn | tail -1
}

for p in fp32 bf16 int8 fp8; do
  for m in superpoint disk alike xfeat dedode dedode-g dad ripe; do run "$m-$p" $m $p 4096; done
  for m in superpoint disk xfeat; do
    run "$m-lightglue-$p" $m-lightglue $p 4096 --overrides components.extractor.preset=$p
  done
  run "loftr-$p" loftr $p 0
done
run sift-fp32 sift fp32 4096
run orb-fp32 orb fp32 4096
for att in sage sage-fp8; do
  run "xfeat-lightglue-bf16-$att" xfeat-lightglue bf16 4096 --overrides components.extractor.preset=bf16 components.matcher.optimize.attention=$att
done
