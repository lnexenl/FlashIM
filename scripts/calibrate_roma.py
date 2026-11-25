"""Calibrate static activation scales for RoMa (v1) with FlashIM's BF16 model.

Runs ``match_dense`` over manifest pairs with observers on every statically quantized layer of the chosen preset
(``--format mxfp8``: block-32 amax exponents; ``--format int8``: per-tensor 99.99th-percentile scales) and writes
a FlashIM calibration table (safetensors).

    python scripts/calibrate_roma.py \
        --manifest data/quantization/calibration_local.jsonl \
        --root . --variant outdoor --max-pairs 16 --format mxfp8 \
        --out flashim/models/roma/calibration/roma_outdoor_fp8_mega.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import flashim
from flashim.models.roma import quant
from flashim.quant.calibration import Calibrator, Int8Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL (or JSON list) with image_a / image_b")
    parser.add_argument("--root", default=".", help="base directory for relative image paths")
    parser.add_argument("--variant", choices=("outdoor", "indoor", "minima"), default="outdoor")
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--format", choices=("mxfp8", "int8"), default="mxfp8")
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

    # Run fold_batchnorm + pad_refiners only (skip fused_bf16_refiners so padded refiner 1x1s remain nn.Conv2d).
    model_name = {"outdoor": "roma", "indoor": "roma-indoor", "minima": "minima-roma"}[args.variant]
    model = flashim.load(
        model_name,
        "bf16",
        overrides={"optimize.passes": ["fold_batchnorm", "pad_refiners"]},
    )
    is_int8 = args.format == "int8"
    scheme_config = quant.QUANT_CONFIGS["int8" if is_int8 else "fp8"]

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            continue
        match = scheme_config.scheme_for(name)
        if match is None:
            continue
        act = match[1].input_activations
        if act is not None and not act.dynamic:
            targets.append(name)

    root = Path(args.root).expanduser()
    with torch.inference_mode(), (Int8Calibrator if is_int8 else Calibrator)(model, targets) as calibrator:
        for index, item in enumerate(pairs, start=1):
            model.match_dense(root / item["image_a"], root / item["image_b"])
            print(f"calibrated {index}/{len(pairs)}", flush=True)

    manifest_sha = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    table = calibrator.table(
        {
            "variant": args.variant,
            "pairs": str(len(pairs)),
            "manifest": Path(args.manifest).name,
            "manifest_sha256": manifest_sha,
        }
    )
    tmp = Path(args.out).with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(args.out)
    n_layers = len(getattr(table, "linear", None) or table.input_scale)
    print(json.dumps({"out": args.out, "pairs": len(pairs), "layers": n_layers}))


if __name__ == "__main__":
    main()
