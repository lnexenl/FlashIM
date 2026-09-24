"""Calibrate static MXFP8 activation scales for RoMaV2 with FlashIM's BF16 model.

Runs ``match`` over manifest pairs with amax observers on every linear targeted by the
static transformer scheme, and writes a FlashIM calibration table (safetensors).

    python scripts/calibrate_romav2.py \
        --manifest ~/work/RoMaV2/data/quantization/calibration_local.jsonl \
        --root ~/work/RoMaV2 --setting mega1500 --max-pairs 8 --out cal.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from flashim.models.romav2 import RoMaV2, RoMaV2Config
from flashim.models.romav2.quant import PRESETS
from flashim.quant.calibration import Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL (or JSON list) with image_a / image_b")
    parser.add_argument("--root", default=".", help="base directory for relative image paths")
    parser.add_argument("--setting", default="mega1500")
    parser.add_argument("--max-pairs", type=int, default=8)
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

    model = RoMaV2.from_pretrained(RoMaV2Config(setting=args.setting))
    scheme_config = PRESETS["fp8-p8"].config
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and scheme_config.scheme_for(name) is not None
    ]
    root = Path(args.root).expanduser()
    with torch.inference_mode(), Calibrator(model, targets) as calibrator:
        for index, item in enumerate(pairs, start=1):
            model.match(root / item["image_a"], root / item["image_b"])
            print(f"calibrated {index}/{len(pairs)}", flush=True)
    manifest_sha = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    table = calibrator.table(
        {
            "setting": args.setting,
            "pairs": str(len(pairs)),
            "manifest": Path(args.manifest).name,
            "manifest_sha256": manifest_sha,
        }
    )
    table.save(args.out)
    print(json.dumps({"out": args.out, "pairs": len(pairs), "linears": len(table.linear)}))


if __name__ == "__main__":
    main()
