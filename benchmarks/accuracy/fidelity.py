"""Fidelity of reduced-precision presets against the same model's fp32 preset (a cheap proxy, NOT the accepted accuracy
criterion — that is task-level AUC, see ``sparse_matrix.sh``).

For each of the first ``--pairs-per-scene`` MegaDepth-1500 pairs of every scene, and each model / preset:

- ``kp_overlap``: fraction of the fp32 keypoints that the preset also detects within 1.5 px (extractors);
- ``desc_cos``: median cosine between the two descriptors of such a keypoint pair;
- ``match_agree``: fraction of the fp32 mutual-NN matches (as coordinate pairs, within 2 px on both sides) that the preset
  reproduces, and ``match_ratio``: its match count over fp32's. For LoFTR (pair-level) the matches are compared directly.

    python benchmarks/accuracy/fidelity.py --models superpoint disk alike xfeat dedode dedode-g dad ripe loftr
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run import MEGA_SCENES

import flashim
from flashim.paths import DATA

PROTOCOL = {  # model -> (resize, top_k), as benchmarks/accuracy/sparse_matrix.sh
    "superpoint": (1600, 4096), "disk": (1600, 4096), "alike": (1600, 4096), "xfeat": (1600, 4096),
    "dedode": (1024, 8192), "dedode-g": (1024, 8192), "dad": (1024, 8192), "ripe": (1024, 4096), "loftr": (840, 0),
}


def pairs(per_scene: int) -> list[tuple[str, str]]:
    root = DATA / "megadepth"
    out = []
    for scene_name in MEGA_SCENES:
        scene = np.load(root / scene_name, allow_pickle=True)
        infos, paths = scene["pair_infos"], scene["image_paths"]
        for i in range(per_scene):
            a, b = infos[i][0]
            out.append((f"{root}/{paths[a]}", f"{root}/{paths[b]}"))
    return out


def near(a: torch.Tensor, b: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row of ``a`` the nearest row of ``b``: (distance, index)."""
    if len(b) == 0:
        return a.new_full((len(a),), float("inf")), torch.zeros(len(a), dtype=torch.long, device=a.device)
    d = torch.cdist(a.float(), b.float(), compute_mode="donot_use_mm_for_euclid_dist")
    return d.min(1).values, d.argmin(1)


def agreement(ref_a, ref_b, a, b, radius: float = 2.0) -> float:
    """Fraction of reference matches ``(ref_a[i], ref_b[i])`` present in ``(a, b)`` within ``radius`` px on both sides."""
    if len(ref_a) == 0:
        return float("nan")
    if len(a) == 0:
        return 0.0
    da = torch.cdist(ref_a.float(), a.float(), compute_mode="donot_use_mm_for_euclid_dist")
    db = torch.cdist(ref_b.float(), b.float(), compute_mode="donot_use_mm_for_euclid_dist")
    return float(((da < radius) & (db < radius)).any(1).float().mean())


def evaluate(model_name: str, presets: list[str], image_pairs: list[tuple[str, str]]) -> list[dict]:
    resize, top_k = PROTOCOL[model_name]
    pair_level = model_name == "loftr"
    reference: dict[str, object] = {}
    rows = []
    for preset in ["fp32", *presets]:
        model = flashim.load(model_name, preset)
        stats: dict[str, list[float]] = {"kp_overlap": [], "desc_cos": [], "match_agree": [], "match_ratio": []}
        for index, (pa, pb) in enumerate(image_pairs):
            if pair_level:
                r = model.match(pa, pb, resize=resize, top_k=top_k or None)
                ours = (r.keypoints_A, r.keypoints_B)
                feats = None
            else:
                fa, fb = (model.extract(p, resize=resize, top_k=top_k) for p in (pa, pb))
                r = model.match_features(fa, fb)
                ours, feats = (r.keypoints_A, r.keypoints_B), (fa, fb)
            if preset == "fp32":
                reference[index] = (ours, feats, len(r.confidence))
                continue
            (ref_a, ref_b), ref_feats, ref_count = reference[index]
            stats["match_agree"].append(agreement(ref_a, ref_b, *ours))
            stats["match_ratio"].append(len(r.confidence) / max(ref_count, 1))
            if feats is not None:
                for f, rf in zip(feats, ref_feats):
                    d, j = near(rf.keypoints[0], f.keypoints[0], 1.5)
                    ok = d < 1.5
                    stats["kp_overlap"].append(float(ok.float().mean()))
                    if ok.any():
                        cos = (rf.descriptors[0][ok].float() * f.descriptors[0][j[ok]].float()).sum(-1)
                        stats["desc_cos"].append(float(cos.median()))
        if preset != "fp32":
            rows.append({"model": model_name, "preset": preset, **{
                k: (round(statistics.fmean(v), 4) if v else None) for k, v in stats.items()
            }})
        del model
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(PROTOCOL))
    parser.add_argument("--presets", nargs="+", default=["bf16", "int8", "fp8"])
    parser.add_argument("--pairs-per-scene", type=int, default=4)
    parser.add_argument("--out", default="results/accuracy/fidelity.jsonl")
    args = parser.parse_args()
    image_pairs = pairs(args.pairs_per_scene)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), open(args.out, "a") as sink:
        for name in args.models:
            for row in evaluate(name, args.presets, image_pairs):
                print(json.dumps(row), flush=True)
                sink.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
