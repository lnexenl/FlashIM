"""Relative-pose accuracy of a FlashIM dense matcher on MegaDepth-1500 / ScanNet-1500.

Protocol (identical to RoMaV2's benchmarks): per pair one ``match``; MegaDepth
draws 5 fresh ``sample(5000)`` sets per pair (intrinsics rescaled to a 1200px long
side), ScanNet samples once and reshuffles 5 times (480px short side, -0.5 px
offset); OpenCV essential-matrix RANSAC with ``0.5px`` normalized threshold;
AUC@5/10/20 of ``max(e_t, e_R)``.

    python benchmarks/accuracy/run.py --dataset mega1500 --preset fp8 --engine compile --seed 17
"""

from __future__ import annotations

import argparse
import json
import logging
import os.path as osp
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import flashim
from flashim.geometry.pose import (
    compute_pose_error,
    compute_relative_pose,
    estimate_pose_cv2_ransac,
    summarize,
)

logger = logging.getLogger(__name__)
ROMAV2_DATA = Path.home() / "work" / "RoMaV2" / "data"
MEGA_SCENES = ("0015_0.1_0.3.npz", "0015_0.3_0.5.npz", "0022_0.1_0.3.npz", "0022_0.3_0.5.npz", "0022_0.5_0.7.npz")


def _pose_errors(model, kpts1, kpts2, K1, K2, R, t, errors: list[float]) -> None:
    norm_threshold = 0.5 / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
    try:
        R_est, t_est, _ = estimate_pose_cv2_ransac(kpts1, kpts2, K1, K2, norm_threshold, conf=0.99999)
        e_t, e_R = compute_pose_error(R_est, t_est[:, 0], R, t)
        errors.append(max(e_t, e_R))
    except Exception as exc:  # degenerate geometry counts as a failure (90 degrees)
        logger.debug("pose estimation failed: %s", exc)
        errors.append(90.0)


def mega1500(model, data_root: Path, max_pairs: int = 0) -> tuple[list[float], int]:
    errors: list[float] = []
    count = 0
    for scene_name in MEGA_SCENES:
        scene = np.load(data_root / scene_name, allow_pickle=True)
        pairs, intrinsics, poses, paths = (scene[k] for k in ("pair_infos", "intrinsics", "poses", "image_paths"))
        indices = range(len(pairs)) if max_pairs <= 0 else range(min(len(pairs), max(1, max_pairs // 5)))
        for i in tqdm(indices, desc=f"mega1500 {scene_name}", leave=False):
            idx1, idx2 = pairs[i][0]
            K1, K2 = intrinsics[idx1].copy(), intrinsics[idx2].copy()
            T1, T2 = poses[idx1], poses[idx2]
            R, t = compute_relative_pose(T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3])
            path_a, path_b = f"{data_root}/{paths[idx1]}", f"{data_root}/{paths[idx2]}"
            preds = model.match(path_a, path_b)
            (w1, h1), (w2, h2) = Image.open(path_a).size, Image.open(path_b).size
            s1, s2 = 1200 / max(w1, h1), 1200 / max(w2, h2)
            w1, h1, w2, h2 = s1 * w1, s1 * h1, s2 * w2, s2 * h2
            K1[:2] *= s1
            K2[:2] *= s2
            for _ in range(5):
                matches, *_ = model.sample(preds, 5000)
                k1, k2 = model.to_pixel_coordinates(matches, h1, w1, h2, w2)
                k1, k2 = k1.cpu().numpy(), k2.cpu().numpy()
                perm = np.random.permutation(len(k1))
                _pose_errors(model, k1[perm], k2[perm], K1, K2, R, t, errors)
            count += 1
    return errors, count


def scannet1500(model, data_root: Path, max_pairs: int = 0) -> tuple[list[float], int]:
    data = np.load(osp.join(data_root, "test.npz"))
    pairs, rel_pose = data["name"], data["rel_pose"]
    if max_pairs > 0:
        pairs, rel_pose = pairs[:max_pairs], rel_pose[:max_pairs]
    errors: list[float] = []
    for i in tqdm(range(len(pairs)), desc="scannet1500", leave=False):
        scene = pairs[i]
        scene_dir = osp.join(data_root, "scans_test", f"scene0{scene[0]}_00")
        path_a = osp.join(scene_dir, "color", f"{scene[2]}.jpg")
        path_b = osp.join(scene_dir, "color", f"{scene[3]}.jpg")
        T = rel_pose[i].reshape(3, 4)
        R, t = T[:3, :3], T[:3, 3]
        with open(osp.join(scene_dir, "intrinsic", "intrinsic_color.txt")) as handle:
            K = np.stack([np.array([float(v) for v in row.split()]) for row in handle.read().split("\n") if row])
        (w1, h1), (w2, h2) = Image.open(path_a).size, Image.open(path_b).size
        preds = model.match(path_a, path_b)
        matches, *_ = model.sample(preds, 5000)
        s1, s2 = 480 / min(w1, h1), 480 / min(w2, h2)
        w1, h1, w2, h2 = s1 * w1, s1 * h1, s2 * w2, s2 * h2
        K1, K2 = K.copy(), K.copy()
        K1[:2] *= s1
        K2[:2] *= s2
        k1, k2 = model.to_pixel_coordinates(matches, h1, w1, h2, w2)
        k1, k2 = k1.cpu().numpy() - 0.5, k2.cpu().numpy() - 0.5
        for _ in range(5):
            perm = np.random.permutation(len(k1))
            k1, k2 = k1[perm], k2[perm]
            _pose_errors(model, k1, k2, K1, K2, R, t, errors)
    return errors, len(pairs)


def _git(cwd: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("mega1500", "scannet1500"), required=True)
    parser.add_argument("--model", default="romav2")
    parser.add_argument("--preset", default="fp8")
    parser.add_argument("--engine", default="compile", choices=("eager", "compile"))
    parser.add_argument("--data-root", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all 1500")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    data_root = Path(args.data_root) if args.data_root else (
        ROMAV2_DATA / "megadepth" if args.dataset == "mega1500" else ROMAV2_DATA / "scannet" / "scans"
    )
    model = flashim.load(args.model, preset=args.preset, setting=args.dataset, engine=args.engine)
    started = time.time()
    run = mega1500 if args.dataset == "mega1500" else scannet1500
    errors, pairs = run(model, data_root, args.max_pairs)
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "preset": args.preset,
        "engine": args.engine,
        "seed": args.seed,
        "pairs": pairs,
        **summarize(errors),
        "seconds": time.time() - started,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "flashim_git": _git(Path(__file__).parent, "rev-parse", "HEAD"),
        "calibration": getattr(model, "calibration_metadata", None),
        "packing": model.quant_report.summary() if getattr(model, "quant_report", None) else None,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
