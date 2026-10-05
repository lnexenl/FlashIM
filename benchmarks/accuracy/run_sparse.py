"""Relative-pose accuracy of a FlashIM sparse matcher on MegaDepth-1500.

Unified sparse protocol (one setting for every model, so results compare across models): per pair one
``match`` at a 1200px long side with 4096 keypoints per image and each model's own matcher, keypoints
mapped back to the *original* images' intrinsics; poselib essential-matrix LO-RANSAC with a 1.0px
threshold (``--estimator poselib --ransac-thr 1.0``); AUC@5/10/20 of ``max(e_t, e_R)``.

The oracles differ (XFeat: 1200px, poselib, 1.0px; RDD: 1600px, OpenCV, 0.5; LoFTR: 840px, OpenCV; DaD:
poselib, 2.0px), so each published number needs its own setting to be reproduced: ``--resize``,
``--estimator``, ``--ransac-thr`` (in pixels of the original image; the OpenCV path normalizes it by the
summed mean focal lengths, so ``--estimator opencv --ransac-thr 0.5`` is RDD's ``0.5`` exactly) and
``--bgr`` (RDD and XFeat feed OpenCV's BGR arrays to the network; FlashIM feeds RGB by default).

    python benchmarks/accuracy/run_sparse.py --model xfeat --preset fp32 --seed 17
    python benchmarks/accuracy/run_sparse.py --model rdd --resize 1600 --estimator opencv --ransac-thr 0.5 --bgr
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import argparse
import json
import logging
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run import MEGA_SCENES, ROMAV2_DATA, PosePool, _git, _read_rgb, prefetch

import flashim
import flashim.registry
from flashim.geometry.pose import (
    compute_pose_error,
    compute_relative_pose,
    estimate_pose_cv2_ransac,
    summarize,
)

logger = logging.getLogger(__name__)


def _poselib_pose(seed: int, kpts1, kpts2, K1, K2, pixel_thr: float, conf: float = 0.99999):
    """Essential-matrix LO-RANSAC (poselib) on pixel keypoints; returns ``(R, t[3, 1])``."""
    import poselib

    def camera(K):
        return {"model": "PINHOLE", "width": 0, "height": 0, "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]}

    options = {"max_epipolar_error": pixel_thr, "success_prob": conf, "seed": seed}
    try:
        pose, _ = poselib.estimate_relative_pose(
            kpts1.astype(np.float64), kpts2.astype(np.float64), camera(K1), camera(K2), options, {}
        )
    except (TypeError, KeyError, ValueError):  # poselib versions without a "seed" RANSAC option
        options.pop("seed")
        pose, _ = poselib.estimate_relative_pose(
            kpts1.astype(np.float64), kpts2.astype(np.float64), camera(K1), camera(K2), options, {}
        )
    return np.asarray(pose.R), np.asarray(pose.t).reshape(3, 1)


def _sparse_pose_error(estimator: str, pixel_thr: float, seed: int, kpts1, kpts2, K1, K2, R, t) -> float:
    if len(kpts1) < 5:
        return 90.0
    try:
        if estimator == "poselib":
            R_est, t_est = _poselib_pose(seed, kpts1, kpts2, K1, K2, pixel_thr)
        else:
            cv2.setRNGSeed(seed)
            norm = pixel_thr / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
            R_est, t_est, _ = estimate_pose_cv2_ransac(kpts1, kpts2, K1, K2, norm, conf=0.99999)
        e_t, e_R = compute_pose_error(R_est, t_est[:, 0], R, t)
        return max(e_t, e_R)
    except Exception as exc:  # degenerate geometry / no model found counts as a failure (90 degrees)
        logger.debug("pose estimation failed: %s", exc)
        return 90.0


class SeededPosePool(PosePool):
    """PosePool with a selectable estimator; RANSAC seeds are offset by ``seed_base``."""

    def __init__(self, workers: int, seed_base: int, estimator: str, pixel_thr: float):
        super().__init__(workers)
        self.seed_base = seed_base
        self.estimator = estimator
        self.pixel_thr = pixel_thr

    def submit(self, *job) -> None:
        self._slots.acquire()
        fut: Future = self._pool.submit(
            _sparse_pose_error, self.estimator, self.pixel_thr, self.seed_base + len(self._futures), *job
        )
        fut.add_done_callback(lambda _: self._slots.release())
        self._futures.append(fut)


class MultiSeedPool:
    """One :class:`SeededPosePool` per RANSAC seed over the same matches (the pipeline itself is deterministic, so the
    seeds only differ in pose estimation): ``errors()`` returns one error list per seed."""

    def __init__(self, workers: int, seeds: list[int], estimator: str, pixel_thr: float):
        per_pool = max(1, workers // len(seeds))
        self.pools = [SeededPosePool(per_pool, seed, estimator, pixel_thr) for seed in seeds]

    def submit(self, *job) -> None:
        for pool in self.pools:
            pool.submit(*job)

    def errors(self) -> list[list[float]]:
        return [pool.errors() for pool in self.pools]


def mega1500_sparse(model, data_root: Path, pool: PosePool, max_pairs: int, loader: dict, match_kwargs: dict, bgr: bool) -> int:
    def jobs():
        for scene_name in MEGA_SCENES:
            scene = np.load(data_root / scene_name, allow_pickle=True)
            pairs, intrinsics, poses, paths = (scene[k] for k in ("pair_infos", "intrinsics", "poses", "image_paths"))
            n = len(pairs) if max_pairs <= 0 else min(len(pairs), max(1, max_pairs // 5))
            for i in range(n):
                idx1, idx2 = pairs[i][0]
                T1, T2 = poses[idx1], poses[idx2]
                R, t = compute_relative_pose(T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3])
                yield (f"{data_root}/{paths[idx1]}", f"{data_root}/{paths[idx2]}", intrinsics[idx1].copy(), intrinsics[idx2].copy(), R, t)

    def read(path):
        image = _read_rgb(path)
        return np.ascontiguousarray(image[..., ::-1]) if bgr else image

    def load(job):
        return (*job, read(job[0]), read(job[1]))

    count = 0
    for *_, K1, K2, R, t, img_a, img_b in tqdm(prefetch(jobs(), load, **loader), desc="mega1500", leave=False):
        result = model.match(img_a, img_b, **match_kwargs)
        pool.submit(result.keypoints_A.cpu().numpy(), result.keypoints_B.cpu().numpy(), K1, K2, R, t)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rdd")
    parser.add_argument("--preset", default="fp8")
    parser.add_argument("--engine", default="eager")
    parser.add_argument("--resize", type=int, default=1200, help="long side fed to the model")
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--estimator", choices=("poselib", "opencv"), default="poselib")
    parser.add_argument("--ransac-thr", type=float, default=1.0, help="inlier threshold in pixels of the original image")
    parser.add_argument("--bgr", action="store_true", help="feed BGR like RDD's / XFeat's own benchmark scripts")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--seed", type=int, default=0, help="RANSAC seed offset")
    parser.add_argument("--seeds", type=int, nargs="*", default=[], help="several RANSAC seeds on the same matches: "
                        "the reported AUC / mAP are their means, per-seed values go to *_per_seed (overrides --seed)")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all 1500")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4), help="RANSAC threads")
    parser.add_argument("--loaders", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--extra", default="", help="JSON dict of extra flashim.load kwargs (legacy keyword API)")
    parser.add_argument("--overrides", nargs="*", default=[], help="dotted config overrides (config API)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    ncpu = os.cpu_count() or 4
    cv2.setNumThreads(ncpu)
    torch.set_num_threads(ncpu)
    torch.backends.cudnn.benchmark = False
    data_root = Path(args.data_root) if args.data_root else ROMAV2_DATA / "megadepth"
    # keyword API for models that have one, unless --overrides are given (it would drop them silently)
    if args.extra or (args.model in flashim.registry.LEGACY_LOADERS and not args.overrides):
        model = flashim.load(args.model, preset=args.preset, engine=args.engine, **(json.loads(args.extra) if args.extra else {}))
    else:
        model = flashim.load(args.model, args.preset, overrides=[f"engine.kind={args.engine}", *args.overrides])
    started = time.time()
    seeds = args.seeds or [args.seed]
    pool = MultiSeedPool(args.workers, seeds, args.estimator, args.ransac_thr)
    pairs = mega1500_sparse(
        model, data_root, pool, args.max_pairs, {"workers": args.loaders, "depth": args.prefetch},
        {"resize": args.resize, "top_k": args.top_k}, args.bgr,
    )
    per_seed = [summarize(errors) for errors in pool.errors()]
    mean = {key: float(np.mean([s[key] for s in per_seed])) for key in per_seed[0]}
    result = {
        "dataset": "mega1500",
        "model": args.model,
        "preset": args.preset,
        "engine": args.engine,
        "resize": args.resize,
        "top_k": args.top_k,
        "estimator": args.estimator,
        "ransac_thr": args.ransac_thr,
        "bgr": args.bgr,
        "seed": seeds[0] if len(seeds) == 1 else None,
        "seeds": seeds,
        "pairs": pairs,
        **mean,
        "auc_5_per_seed": [s["auc_5"] for s in per_seed],
        "auc_5_std": float(np.std([s["auc_5"] for s in per_seed])),
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
