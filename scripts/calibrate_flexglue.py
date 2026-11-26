"""Calibrate static activation scales for FlexGlue's transformer linears (MXFP8 or INT8).

Runs RDD (``--rdd-preset``, default ``bf16``) at ``--top-k`` keypoints on the MegaDepth image
pairs of ``--manifest``, then FlexGlue in BF16 with observers on every linear its FP8/INT8
presets target (``flashim.models.flexglue.quant.TRANSFORMER_LINEARS``). Both the keypoint path
(``match_keypoints``) and the dense sampled path (``match``) are exercised, so every targeted
linear sees activations of both shapes.

    python scripts/calibrate_flexglue.py --format fp8 --manifest data/quantization/calibration_megadepth.jsonl \
        --max-pairs 16 --out flashim/models/flexglue/calibration/flexglue_fp8_mega.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import flashim
from flashim.models.flexglue.quant import PRESETS
from flashim.quant.calibration import Calibrator, Int8Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=("fp8", "int8"), required=True)
    parser.add_argument("--manifest", required=True,
                        help="JSONL (or JSON list) with image_a / image_b, e.g. data/quantization/calibration_megadepth.jsonl")
    parser.add_argument("--root", default="data/quantization/megadepth_images",
                        help="base directory for the manifest's relative image paths")
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--rdd-preset", default="bf16", help="extractor preset (the features FlexGlue sees in deployment)")
    parser.add_argument("--percentile", type=float, default=0.9999, help="INT8 observer quantile of |x|")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    text = manifest.read_text()
    try:
        raw = json.loads(text)
        pairs = raw["pairs"] if isinstance(raw, dict) else raw
    except json.JSONDecodeError:
        pairs = [json.loads(line) for line in text.splitlines() if line.strip()]
    root = Path(args.root).expanduser()
    paths = [
        (root / item.get("image_a_relative", item["image_a"]), root / item.get("image_b_relative", item["image_b"]))
        for item in pairs
    ]
    manifest_meta = {"manifest": manifest.name, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    paths = paths[: args.max_pairs] if args.max_pairs else paths
    if not paths:
        raise ValueError("empty calibration image list")

    rdd = flashim.load("rdd", preset=args.rdd_preset, device="cuda")
    model = flashim.load("flexglue", "bf16", device="cuda")
    config = PRESETS[args.format].config
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and config.scheme_for(name)
    ]
    print(f"observing {len(targets)} linears over {len(paths)} pairs", flush=True)
    observer = Int8Calibrator(model, targets, percentile=args.percentile) if args.format == "int8" else Calibrator(model, targets)
    with torch.inference_mode(), observer:
        for index, (path_a, path_b) in enumerate(paths, start=1):
            feats_a, feats_b = (rdd.extract(str(path), resize=None, top_k=args.top_k) for path in (path_a, path_b))
            kpts_a, kpts_b = (f.keypoints.reshape(-1, 2) for f in (feats_a, feats_b))
            model.match_keypoints(str(path_a), kpts_a, str(path_b), kpts_b)
            model.match(str(path_a), str(path_b), resize=None, max_matches=5000, threshold=None)
            print(f"calibrated {index}/{len(paths)}", flush=True)
    table = observer.table(
        {
            "model": "flexglue",
            "extractor": "rdd",
            "rdd_preset": args.rdd_preset,
            "pairs": str(len(paths)),
            **manifest_meta,
        }
    )
    tmp = Path(args.out).with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(args.out)
    print(json.dumps({"out": args.out, "pairs": len(paths), "layers": len(targets)}))


if __name__ == "__main__":
    main()
