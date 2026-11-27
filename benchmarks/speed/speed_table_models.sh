#!/usr/bin/env bash
# Latency of the models that speed_table.sh does not cover (LoMa, SuperGlue / OmniGlue pipelines, the dense and
# semi-dense matchers, FlexGlue), x preset x engine, one configuration per process; resumable like speed_table.sh.
#
#   setsid nohup benchmarks/speed/speed_table_models.sh >> results/speed/table/run_models.log 2>&1 &
#   .venv/bin/python benchmarks/speed/speed_table.py               # -> results/speed/table/table.md
#
# Feature-level pipelines (benchmarks/speed/extractor.py): loma-* at 1024 px with 4096 keypoints, superpoint-superglue
# and omniglue at 1024 px with 1024 keypoints (the accuracy protocol of each model). Pair-level matchers
# (extractor.py, `match` of the pair): at the model's own default resolution; dense ones keep 5000 sampled
# matches, semi-dense ones all their matches. FlexGlue: benchmarks/speed/flexglue.py (2048 RDD keypoints for
# `match_keypoints`, 5000 samples for the dense `match()`, 512x512 descriptor).
# Presets a model does not have are not run (they would only repeat the preset they fall back to).
# Re-measure only what a change touches: ONLY="loma-g flexglue" RERUN=1 benchmarks/speed/speed_table_models.sh
# (ONLY: space-separated model names as in the tags; RERUN=1 deletes their old results first).
cd "$(dirname "$0")/../.."
OUT_DIR=${OUT_DIR:-results/speed/table}
mkdir -p "$OUT_DIR"
out=$OUT_DIR/models.jsonl
touch "$out"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/flashim/inductor}

want() {  # is this model selected (all of them when ONLY is empty)?
  [[ -z ${ONLY:-} || " $ONLY " == *" $1 "* ]]
}
if [[ -n ${ONLY:-} && -n ${RERUN:-} ]]; then  # forget the old results of the selected models
  for m in $ONLY; do
    grep -vF "\"tag\": \"$m|" "$out" > "$out.tmp" || true
    mv "$out.tmp" "$out"
  done
fi

run() {  # model preset engine resize top_k
  local model=$1 preset=$2 engine=$3 resize=$4 topk=$5
  local tag="$model|$preset|$engine" line
  want "$model" || return 0
  grep -qF "\"tag\": \"$tag\"" "$out" && return 0
  echo "== $tag"
  line=$(.venv/bin/python benchmarks/speed/extractor.py --model "$model" --preset "$preset" --engine "$engine" \
    --resize "$resize" --top-k "$topk" --tag "$tag" 2> "$OUT_DIR/${tag//|/_}.err" | tail -1)
  if [[ $line == "{"* ]]; then echo "$line" >> "$out"; else echo "FAILED $tag"; fi
}

flexglue() {  # preset engine
  local tag="flexglue|$1|$2" line
  want flexglue || return 0
  grep -qF "\"tag\": \"$tag\"" "$out" && return 0
  echo "== $tag"
  line=$(.venv/bin/python benchmarks/speed/flexglue.py --preset "$1" --engine "$2" --tag "$tag" \
    2> "$OUT_DIR/${tag//|/_}.err" | tail -1)
  if [[ $line == "{"* ]]; then echo "$line" >> "$out"; else echo "FAILED $tag"; fi
}

for engine in eager compile; do
  for p in fp32 bf16 int8 fp8; do
    for m in loma-b128 loma-b loma-r loma-l loma-g; do run $m $p $engine 1024 4096; done
    for m in roma minima-roma tiny-roma mast3r dust3r ufm; do run $m $p $engine 0 5000; done
    for m in eloftr xoftr minima-loftr; do run $m $p $engine 0 0; done
    run traqpoint $p $engine 1200 4096
    flexglue $p $engine
  done
  for p in fp32 bf16 int8; do
    run superpoint-superglue $p $engine 1024 1024
    run omniglue $p $engine 1024 1024
  done
done
echo done
