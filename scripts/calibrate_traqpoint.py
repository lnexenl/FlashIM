"""Calibrate static MXFP8 and INT8 activation scales for TraqPoint with FlashIM's BF16 model.

Runs ``extract`` over the images of manifest pairs with observers on every linear in the backbone ConvNeXt MLP and
deformable transformer (``CALIBRATED_FP8`` / ``CALIBRATED_INT8``), and writes a FlashIM calibration table.

    python scripts/calibrate_traqpoint.py --format fp8 \\
        --manifest data/quantization/calibration_megadepth.jsonl --root data/quantization/megadepth_images \\
        --max-pairs 16 --out flashim/models/traqpoint/calibration/traqpoint_fp8_mega.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import flashim
from flashim.models.traqpoint.quant import CALIBRATED_FP8, CALIBRATED_INT8
from flashim.quant.calibration import Calibrator, Int8Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=("fp8", "int8"), required=True)
    parser.add_argument("--manifest", required=True, help="JSONL (or JSON list) with image_a / image_b")
    parser.add_argument("--root", default="data/quantization/megadepth_images", help="base directory for relative paths")
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--resize", type=int, default=1600, help="long side of the working resolution")
    parser.add_argument("--percentile", type=float, default=0.9999, help="INT8 observer quantile of |x|")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    text = manifest.read_text()
    try:
        raw = json.loads(text)
        pairs = raw["pairs"] if isinstance(raw, dict) else raw
    except json.JSONDecodeError:
        pairs = [json.loads(line) for line in text.splitlines() if line.strip()]
    pairs = pairs[: args.max_pairs] if args.max_pairs else pairs
    if not pairs:
        raise ValueError("empty calibration manifest")

    model = flashim.load("traqpoint", "bf16", device="cuda", overrides=([] if not args.checkpoint else [f"checkpoint={args.checkpoint}"]))
    scheme_config = CALIBRATED_INT8 if args.format == "int8" else CALIBRATED_FP8
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and scheme_config.scheme_for(name) is not None
    ]
    print(f"observing {len(targets)} linears ({args.format}) over {len(pairs)} pairs", flush=True)
    observer = (
        Int8Calibrator(model, targets, percentile=args.percentile)
        if args.format == "int8"
        else Calibrator(model, targets)
    )
    root = Path(args.root).expanduser()
    with torch.inference_mode(), observer:
        for index, item in enumerate(pairs, start=1):
            for key_rel, key_plain in (("image_a_relative", "image_a"), ("image_b_relative", "image_b")):
                path = root / item.get(key_rel, item[key_plain])
                model.extract(str(path), resize=args.resize)
            print(f"calibrated {index}/{len(pairs)}", flush=True)
    table = observer.table(
        {
            "model": "traqpoint",
            "resize": str(args.resize),
            "pairs": str(len(pairs)),
            "manifest": manifest.name,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(out_path)
    print(json.dumps({"out": str(out_path), "pairs": len(pairs), "layers": len(targets)}))


if __name__ == "__main__":
    main()
