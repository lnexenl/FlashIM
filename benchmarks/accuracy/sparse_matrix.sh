#!/usr/bin/env bash
# MegaDepth-1500 accuracy of the sparse / semi-dense matchers added after RDD, in fp32 / bf16 / int8 / fp8
# (resumable: existing result files are skipped).
#
#   setsid nohup benchmarks/accuracy/sparse_matrix.sh [model-filter ...] > results/accuracy/sparse_matrix.log 2>&1 &
#
# Unified protocol (comparable across models): every model at a 1200 px long side, 4096 keypoints, poselib LO-RANSAC at 1.0 px
# (run_sparse.py defaults), one run per configuration (the seed only offsets RANSAC). Per-model exceptions: loftr is semi-dense (all coarse matches above 0.2, no keypoint
# budget), dedode / dedode-g / dad keep dual-softmax matching; the *oracle* protocols (XFeat 1200 px poselib 1.0, RDD 1600 px
# OpenCV 0.5, LoFTR 840 px OpenCV, DaD poselib 2.0) are reproduced with run_sparse.py's --resize / --estimator / --ransac-thr / --bgr.
#   sift, orb: ratio test 0.8 (fp32 only: no weights)
# In the *-lightglue pipelines the extractor and the matcher run in the same precision.
# SEEDS (default "17 29 42"): RANSAC seeds evaluated on the same matches (AUC = their mean). CAL_DIR: directory of alternative
# calibration tables (scripts/calibrate_extractor.py --out-dir, lightglue_<extractor>_<fmt>_mega.safetensors); only the
# int8 / fp8 rows run then.
# Results go to results/accuracy/unified/ (the reports read it; results/accuracy/sparse_*.json are the older per-model protocols).
cd "$(dirname "$0")/../.."
OUT_DIR=${OUT_DIR:-results/accuracy/unified}
SEEDS=${SEEDS:-17 29 42}
CAL_DIR=${CAL_DIR:-}
mkdir -p "$OUT_DIR"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/flashim/inductor}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
filter=("$@")

wanted() {  # model
  [ ${#filter[@]} -eq 0 ] && return 0
  for f in "${filter[@]}"; do [ "$f" = "$1" ] && return 0; done
  return 1
}

run() {  # model preset top_k [override ...]
  local model=$1 preset=$2 topk=$3; shift 3
  wanted "$model" || return 0
  local out=$OUT_DIR/sparse_${model}_${preset}.json
  [ -f "$out" ] && return 0
  echo "== $model $preset"
  # shellcheck disable=SC2086
  .venv/bin/python benchmarks/accuracy/run_sparse.py --model "$model" --preset "$preset" --top-k "$topk" --seeds $SEEDS \
    --overrides "$@" --out "$out.tmp" > "$OUT_DIR/sparse_${model}_${preset}.log" 2>&1 \
    && mv "$out.tmp" "$out" || echo "FAILED $model $preset (see $OUT_DIR/sparse_${model}_${preset}.log)"
}

cal() {  # model preset -> calibration override of a single model (nothing without CAL_DIR)
  [ -n "$CAL_DIR" ] && echo "precision.calibration=$CAL_DIR/$1_$2.safetensors"
}

pipeline() {  # extractor preset -> overrides of an <extractor>-lightglue pipeline
  echo "components.extractor.preset=$2"
  [ -n "$CAL_DIR" ] && case $2 in int8|fp8)
    echo "components.extractor.precision.calibration=$CAL_DIR/$1_$2.safetensors"
    echo "components.matcher.precision.calibration=$CAL_DIR/lightglue_$1_$2_mega.safetensors";;
  esac
}

presets="fp32 bf16 int8 fp8"
[ -n "$CAL_DIR" ] && presets="int8 fp8"
for p in $presets; do
  for m in superpoint disk alike xfeat dedode dedode-g dad ripe; do run $m $p 4096 $(cal $m $p); done
  for m in superpoint disk xfeat; do run $m-lightglue $p 4096 $(pipeline $m $p); done
  run loftr $p 0 $(cal loftr $p)
done
[ -n "$CAL_DIR" ] && exit 0
run sift fp32 4096
run orb  fp32 4096
