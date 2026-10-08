"""Calibrate static activation scales for a sparse extractor's INT8 / MXFP8 convolutions and linears.

Runs the model's unquantized ``bf16`` preset over the images of manifest pairs (MegaDepth calibration split, disjoint from
MegaDepth-1500) with observers on every layer its int8 / fp8 quant config targets, and writes the table the spec's
``calibrations`` entry points to.

    python scripts/calibrate_extractor.py --model superpoint --format fp8 --format int8 \\
        --manifest data/quantization/calibration_local.jsonl --root . --max-pairs 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

import flashim
from flashim.build import get_spec
from flashim.quant.calibration import Calibrator, Int8Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", action="append", choices=("fp8", "int8"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--resize", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--percentile", type=float, default=0.9999)
    parser.add_argument("--out-dir", default="", help="default: the spec's calibration path")
    parser.add_argument("--extra", default="", help="JSON dict of config overrides (e.g. arch)")
    args = parser.parse_args()

    text = Path(args.manifest).expanduser().read_text()
    try:
        raw = json.loads(text)
        pairs = raw["pairs"] if isinstance(raw, dict) else raw
    except json.JSONDecodeError:
        pairs = [json.loads(line) for line in text.splitlines() if line.strip()]
    pairs = pairs[: args.max_pairs]
    root = Path(args.root).expanduser()
    spec = get_spec(args.model)
    extra = json.loads(args.extra) if args.extra else {}
    model = flashim.build(args.model, "bf16", config=extra)
    for fmt in args.format:
        config = spec.quant_configs[fmt]
        targets = [
            name for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and config.scheme_for(name) and getattr(m, "groups", 1) == 1
        ]
        observer = Int8Calibrator(model, targets, percentile=args.percentile) if fmt == "int8" else Calibrator(model, targets)
        with torch.inference_mode(), observer:
            for index, item in enumerate(pairs, start=1):
                if hasattr(model, "extract"):
                    for key in ("image_a", "image_b"):
                        model.extract(root / item[key], resize=args.resize, top_k=args.top_k)
                else:  # pair-level matchers (LoFTR)
                    model.match(root / item["image_a"], root / item["image_b"], resize=args.resize)
                print(f"[{fmt}] calibrated {index}/{len(pairs)}", flush=True)
        table = observer.table({
            "model": args.model, "resize": str(args.resize), "top_k": str(args.top_k), "pairs": str(len(pairs)),
            "manifest": Path(args.manifest).name,
            "manifest_sha256": hashlib.sha256(Path(args.manifest).expanduser().read_bytes()).hexdigest(),
        })
        out = Path(args.out_dir) / f"{args.model}_{fmt}.safetensors" if args.out_dir else spec.calibrations[fmt]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.safetensors")
        table.save(tmp)
        tmp.replace(out)
        print(json.dumps({"out": str(out), "layers": len(targets)}))


if __name__ == "__main__":
    main()
