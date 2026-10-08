"""Calibrate static activation scales for LightGlue's transformer linears (MXFP8 or INT8).

Runs RDD (``--rdd-preset``, default ``fp8``: the features LightGlue sees in deployment) on the images of manifest
pairs, then LightGlue in BF16 with observers on every linear its FP8/INT8 presets target.

    python scripts/calibrate_lightglue.py --format fp8 \\
        --manifest data/quantization/calibration_local.jsonl \\
        --root . --max-pairs 32 --out flashim/models/lightglue/calibration/lightglue_fp8_mega.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import flashim
from flashim.models.lightglue.quant import PRESETS
from flashim.quant.calibration import Calibrator, Int8Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=("fp8", "int8"), required=True)
    parser.add_argument("--manifest", required=True, help="JSONL (or JSON list) with image_a / image_b")
    parser.add_argument("--root", default=".", help="base directory for relative image paths")
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--resize", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--rdd-preset", default=None, help="extractor preset (default: fp8 for rdd, fp32 otherwise)")
    parser.add_argument("--extractor", default="rdd", help="rdd | aliked | raco-aliked | superpoint | disk | xfeat (LightGlue variant = extractor)")
    parser.add_argument("--percentile", type=float, default=0.9999, help="INT8 observer quantile of |x|")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    text = Path(args.manifest).read_text()
    try:
        raw = json.loads(text)
        pairs = raw["pairs"] if isinstance(raw, dict) else raw
    except json.JSONDecodeError:
        pairs = [json.loads(line) for line in text.splitlines() if line.strip()]
    pairs = pairs[: args.max_pairs] if args.max_pairs else pairs
    if not pairs:
        raise ValueError("empty calibration manifest")

    rdd = flashim.load(args.extractor, preset=args.rdd_preset or ("fp8" if args.extractor == "rdd" else "fp32"))
    model = flashim.build("lightglue", "bf16", config={"variant": args.extractor, "checkpoint": args.checkpoint})
    config = PRESETS[args.format].config
    targets = [
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and config.scheme_for(name)
    ]
    observer = Int8Calibrator(model, targets, percentile=args.percentile) if args.format == "int8" else Calibrator(model, targets)
    root = Path(args.root).expanduser()
    with torch.inference_mode(), observer:
        for index, item in enumerate(pairs, start=1):
            a, b = (rdd.extract(root / item[key], resize=args.resize, top_k=args.top_k) for key in ("image_a", "image_b"))
            model.match_features(a, b)
            print(f"calibrated {index}/{len(pairs)}", flush=True)
    table = observer.table(
        {
            "model": "lightglue",
            "extractor": args.extractor,
            "rdd_preset": str(args.rdd_preset),
            "resize": str(args.resize),
            "pairs": str(len(pairs)),
            "manifest": Path(args.manifest).name,
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        }
    )
    tmp = Path(args.out).with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(args.out)
    print(json.dumps({"out": args.out, "pairs": len(pairs), "layers": len(targets)}))


if __name__ == "__main__":
    main()
