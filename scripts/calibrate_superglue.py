"""Calibrate static activation scales (and SmoothQuant factors) for SuperGlue's GNN linears."""

from __future__ import annotations

import argparse
import hashlib
import json
from fnmatch import fnmatchcase
from pathlib import Path

import torch

import flashim
from flashim.models.superglue.quant import PRESETS, ROTATED_INT8_SMOOTHING
from flashim.quant.calibration import Calibrator, Int8Calibrator, SmoothingCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("int8", "fp8"), required=True)
    parser.add_argument("--variant", choices=("outdoor", "indoor"), default="outdoor")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--resize", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--percentile", type=float, default=0.9999)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    text = Path(args.manifest).read_text()
    pairs = [json.loads(line) for line in text.splitlines() if line.strip()][: args.max_pairs]
    root = Path(args.root).expanduser()

    extractor = flashim.load(
        "superpoint",
        "bf16",
        config={"arch": {"nms_radius": 4, "detection_threshold": 0.005, "remove_borders": 4}},
    )
    passes = ["fold_batchnorm", "rotate_residual_stream", "pack_qkv"]
    model = flashim.load(
        "superglue",
        "bf16",
        config={"variant": args.variant, "optimize": {"passes": passes}},
    )
    qcfg = PRESETS[args.format].config
    targets = [
        name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and qcfg.scheme_for(name)
    ]

    with torch.inference_mode():
        feats = [
            (
                extractor.extract(root / item["image_a"], resize=args.resize, top_k=args.top_k),
                extractor.extract(root / item["image_b"], resize=args.resize, top_k=args.top_k),
            )
            for item in pairs
        ]

    def run(observer) -> None:
        with torch.inference_mode(), observer:
            for fa, fb in feats:
                model.match_features(fa, fb)

    smooth = None
    if args.format == "int8":
        alphas = {
            name: alpha
            for name in targets
            for glob, alpha in ROTATED_INT8_SMOOTHING.items()
            if fnmatchcase(name, glob)
        }
        smoothing = SmoothingCalibrator(model, alphas)
        run(smoothing)
        smooth = smoothing.factors()
        observer = Int8Calibrator(model, targets, percentile=args.percentile, smooth=smooth)
    else:
        observer = Calibrator(model, targets)
    run(observer)

    table = observer.table(
        {
            "model": "superglue",
            "variant": args.variant,
            "extractor": "superpoint",
            "resize": str(args.resize),
            "top_k": str(args.top_k),
            "pairs": str(len(pairs)),
            "manifest": Path(args.manifest).name,
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "passes": ",".join(passes),
            **({"smoothing": json.dumps(ROTATED_INT8_SMOOTHING)} if smooth else {}),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(out)
    print(json.dumps({"out": str(out), "variant": args.variant, "format": args.format, "layers": len(targets)}))


if __name__ == "__main__":
    main()
