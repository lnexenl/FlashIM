"""Calibrate static activation scales for RoMaV2 with FlashIM's BF16 model.

Runs ``match`` over manifest pairs with observers on every layer targeted by the static
transformer scheme (``--format mxfp8``: block-32 amax exponents; ``--format int8``:
per-tensor 99.99th-percentile scales, RoMaV2's ``percentile`` observer) and writes a
FlashIM calibration table (safetensors).

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
from flashim.quant.calibration import CalibrationTable, Calibrator, Int8CalibrationTable, Int8Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL (or JSON list) with image_a / image_b")
    parser.add_argument("--root", default=".", help="base directory for relative image paths")
    parser.add_argument("--setting", default="mega1500")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--format", choices=("mxfp8", "int8"), default="mxfp8")
    parser.add_argument("--preset", help="preset whose targets are calibrated (default: int8 / fp8-p8)")
    parser.add_argument("--merge", action="store_true", help="keep entries already in --out; calibrate the rest")
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
    int8 = args.format == "int8"
    scheme_config = PRESETS[args.preset or ("int8" if int8 else "fp8-p8")].config

    def eligible(module) -> bool:
        conv1x1 = isinstance(module, torch.nn.Conv2d) and module.kernel_size == (1, 1) and module.groups == 1
        return isinstance(module, torch.nn.Linear) or (int8 and conv1x1)

    targets = [
        name for name, module in model.named_modules() if eligible(module) and scheme_config.scheme_for(name) is not None
    ]
    existing = None
    if args.merge and Path(args.out).exists():
        existing = Int8CalibrationTable.load(args.out) if int8 else CalibrationTable.load(args.out)
        targets = [name for name in targets if name not in existing]
        if not targets:
            print(json.dumps({"out": args.out, "note": "nothing to calibrate"}))
            return
    root = Path(args.root).expanduser()
    with torch.inference_mode(), (Int8Calibrator if int8 else Calibrator)(model, targets) as calibrator:
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
    if existing is not None:
        merged = {**(existing.input_scale if int8 else existing.linear), **(table.input_scale if int8 else table.linear)}
        table = (Int8CalibrationTable(merged, table.metadata) if int8 else CalibrationTable(merged, table.metadata))
    tmp = Path(args.out).with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(args.out)
    print(json.dumps({"out": args.out, "pairs": len(pairs), "layers": len(getattr(table, "linear", None) or table.input_scale)}))


if __name__ == "__main__":
    main()
