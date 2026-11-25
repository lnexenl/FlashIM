"""Calibrate static MXFP8 activation scales for MASt3R on the MegaDepth calibration manifest.

Applies the exact residual-stream rotation (:func:`flashim.models.mast3r.quant.rotate_mast3r`) before observing
every linear targeted by ``FP8_CONFIG``'s static MXFP8 scheme in BF16, and writes the table to
``flashim/models/mast3r/calibration/mast3r_fp8.safetensors`` (or ``--out``).

    python scripts/calibrate_mast3r.py \
        --manifest data/quantization/calibration_local.jsonl --root . --max-pairs 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from flashim.models.mast3r import quant
from flashim.models.mast3r.model import MASt3R, prepare_image
from flashim.models.romav2.model import load_image
from flashim.quant.calibration import Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL (or JSON list) with image_a / image_b")
    parser.add_argument("--root", default=".", help="base directory for relative image paths")
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--out", default=str(quant.FP8_CALIBRATION))
    args = parser.parse_args()

    text = Path(args.manifest).expanduser().read_text()
    try:
        raw = json.loads(text)
        pairs = raw["pairs"] if isinstance(raw, dict) else raw
    except json.JSONDecodeError:
        pairs = [json.loads(line) for line in text.splitlines() if line.strip()]
    pairs = pairs[: args.max_pairs] if args.max_pairs else pairs
    if not pairs:
        raise ValueError("empty calibration manifest")

    model = MASt3R.from_pretrained()
    quant.rotate_mast3r(model)
    model.set_compute_dtype(torch.bfloat16)

    targets = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and (match := quant.FP8_CONFIG.scheme_for(name)) is not None
        and match[1].weights.type == "float"
        and match[1].input_activations is not None
        and not match[1].input_activations.dynamic
    ]
    if not targets:
        raise RuntimeError("FP8_CONFIG has no static MXFP8 targets to calibrate")

    root = Path(args.root).expanduser()
    with torch.inference_mode(), Calibrator(model, targets) as calibrator:
        for index, item in enumerate(pairs, start=1):
            im1, _ = prepare_image(load_image(root / item["image_a"], model.device, decoder="pil"), resize=args.resize)
            im2, _ = prepare_image(load_image(root / item["image_b"], model.device, decoder="pil"), resize=args.resize)
            model._forward_descriptors_eager(im1, im2)
            print(f"[fp8] calibrated {index}/{len(pairs)}", flush=True)

    table = calibrator.table(
        {
            "model": "mast3r",
            "resize": str(args.resize),
            "pairs": str(len(pairs)),
            "rotation": "Q1024_Q768_H64",
            "manifest": Path(args.manifest).name,
            "manifest_sha256": hashlib.sha256(Path(args.manifest).expanduser().read_bytes()).hexdigest(),
        }
    )
    out = Path(args.out)
    tmp = out.with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(out)
    print(json.dumps({"out": str(out), "pairs": len(pairs), "layers": len(table.linear)}))


if __name__ == "__main__":
    main()
