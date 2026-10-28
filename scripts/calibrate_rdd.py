"""Calibrate static MXFP8 activation scales for RDD with FlashIM's BF16 model.

Runs ``extract`` over the images of manifest pairs with observers on every linear that any FP8 preset
targets, and writes a FlashIM calibration table (safetensors, block-32 amax exponents).

    python scripts/calibrate_rdd.py \\
        --manifest data/quantization/calibration_local.jsonl \\
        --root . --max-pairs 16 --out flashim/models/rdd/calibration/rdd_fp8_mega.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from flashim.models.rdd import RDD
from flashim.models.rdd.quant import CALIBRATED
from flashim.quant.calibration import Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL (or JSON list) with image_a / image_b")
    parser.add_argument("--root", default=".", help="base directory for relative image paths")
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--resize", type=int, default=1600, help="long side of the working resolution")
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

    model = RDD.from_pretrained(checkpoint=args.checkpoint)
    model.set_compute_dtype(torch.bfloat16)
    scheme_config = CALIBRATED
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and scheme_config.scheme_for(name) is not None
    ]
    root = Path(args.root).expanduser()
    with torch.inference_mode(), Calibrator(model, targets) as calibrator:
        for index, item in enumerate(pairs, start=1):
            for key in ("image_a", "image_b"):
                model.extract(root / item[key], resize=args.resize)
            print(f"calibrated {index}/{len(pairs)}", flush=True)
    table = calibrator.table(
        {
            "model": "rdd",
            "resize": str(args.resize),
            "pairs": str(len(pairs)),
            "manifest": Path(args.manifest).name,
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        }
    )
    tmp = Path(args.out).with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(args.out)
    print(json.dumps({"out": args.out, "pairs": len(pairs), "layers": len(table.linear)}))


if __name__ == "__main__":
    main()
