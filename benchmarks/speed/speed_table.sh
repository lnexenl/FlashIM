#!/usr/bin/env bash
# Latency of every model x preset x engine on this GPU, one configuration per process; resumable (a
# configuration with a result is skipped, a failed one is retried), so rerun it after an interruption.
#
#   setsid nohup benchmarks/speed/speed_table.sh >> results/speed/table/run.log 2>&1 &   # ~3.5 h, RTX 5070 Ti
#   .venv/bin/python benchmarks/speed/speed_table.py               # -> results/speed/table/table.md
#
# Sparse / semi-dense models: benchmarks/speed/extractor.py, the unified protocol of sparse_matrix.sh (one
# MegaDepth image at 1200 px long side, 4096 keypoints; *-lightglue: extract + LightGlue match_features;
# loftr: match of the pair, all coarse matches). RoMaV2: benchmarks/speed/forward.py (mega1500 setting,
# preloaded bidirectional forward); its compiled engines run in two rounds (they vary 15-25% between
# processes) and the table keeps the faster one.
cd "$(dirname "$0")/../.."
OUT_DIR=${OUT_DIR:-results/speed/table}
mkdir -p "$OUT_DIR"
out=$OUT_DIR/sparse.jsonl
touch "$out"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/flashim/inductor}

sparse() {  # model preset engine [extra extractor.py args...]
  local model=$1 preset=$2 engine=$3; shift 3
  local tag="$model|$preset|$engine" line
  grep -qF "\"tag\": \"$tag\"" "$out" && return 0
  echo "== $tag"
  line=$(.venv/bin/python benchmarks/speed/extractor.py --model "$model" --preset "$preset" \
    --engine "$engine" --resize 1200 --top-k 4096 --tag "$tag" "$@" 2> "$OUT_DIR/${tag//|/_}.err" | tail -1)
  if [[ $line == "{"* ]]; then echo "$line" >> "$out"; else echo "FAILED $tag"; fi
}

romav2() {  # preset engine round
  local file="$OUT_DIR/romav2_$1_$2_r$3.json"
  [ -s "$file" ] && return 0
  echo "== romav2|$1|$2|r$3"
  .venv/bin/python benchmarks/speed/forward.py --preset "$1" --engine "$2" --out "$file" > /dev/null \
    2> "$OUT_DIR/romav2_$1_$2_r$3.err" || echo "FAILED romav2|$1|$2|r$3"
}

extractor_preset() {  # pipeline extractor, LightGlue preset -> the extractor's preset (same, else bf16)
  case "$1:$2" in
    aliked:int8 | aliked:fp8 | raco-aliked:int8 | raco-aliked:fp8 | rdd:int8) echo bf16 ;;
    *) echo "$2" ;;
  esac
}

for engine in eager compile; do
  for p in fp32 bf16 int8 fp8; do
    for m in superpoint disk alike xfeat dedode dedode-g dad ripe; do sparse $m $p $engine; done
    sparse loftr $p $engine --top-k 0
    for m in superpoint disk xfeat aliked raco-aliked rdd; do
      sparse "$m-lightglue" $p $engine --overrides "components.extractor.preset=$(extractor_preset $m $p)"
    done
  done
  for p in fp32 bf16; do sparse aliked $p $engine; sparse raco-aliked $p $engine; done
  for p in fp32 bf16 fp8 fp8-backbone fp8-full; do sparse rdd $p $engine; done
done
sparse sift fp32 eager
sparse orb fp32 eager

ROMAV2_PRESETS="bf16 int8 fp8"
for p in $ROMAV2_PRESETS; do romav2 "$p" eager 1; done
for round in 1 2; do
  for engine in compile compile-full; do
    for p in $ROMAV2_PRESETS; do romav2 "$p" $engine $round; done
  done
done
echo done
