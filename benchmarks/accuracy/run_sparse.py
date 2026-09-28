"""Relative-pose accuracy of a FlashIM sparse matcher (RDD) on MegaDepth-1500.

Protocol (RDD's ``benchmarks/mega_1500.py``): per pair one ``match`` at a 1600px long side with 4096
keypoints per image and mutual dual-softmax matches (threshold 0.01), on the *original* images'
intrinsics; OpenCV essential-matrix RANSAC with a ``0.5px`` normalized threshold; AUC@5/10/20 of
``max(e_t, e_R)``. RDD's script feeds OpenCV's BGR arrays to the network; FlashIM feeds RGB by default
(``--bgr`` reproduces the reference protocol).

    python benchmarks/accuracy/run_sparse.py --preset fp8 --seed 0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import argparse
import json
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run import MEGA_SCENES, ROMAV2_DATA, PosePool, _git, _pose_error, _read_rgb, prefetch

import flashim
from flashim.geometry.pose import compute_relative_pose, summarize


class SeededPosePool(PosePool):
    """PosePool whose RANSAC seeds are offset by ``seed_base`` (the pipeline itself is deterministic)."""

    def __init__(self, workers: int, seed_base: int):
        super().__init__(workers)
        self.seed_base = seed_base

    def submit(self, *job) -> None:
        self._slots.acquire()
        fut: Future = self._pool.submit(_pose_error, self.seed_base + len(self._futures), *job)
        fut.add_done_callback(lambda _: self._slots.release())
        self._futures.append(fut)


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
    parser.add_argument("--resize", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--bgr", action="store_true", help="feed BGR like RDD's own benchmark script")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--seed", type=int, default=0, help="RANSAC seed offset")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all 1500")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4), help="RANSAC threads")
    parser.add_argument("--loaders", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--extra", default="", help="JSON dict of extra flashim.load kwargs")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    ncpu = os.cpu_count() or 4
    cv2.setNumThreads(ncpu)
    torch.set_num_threads(ncpu)
    torch.backends.cudnn.benchmark = False
    data_root = Path(args.data_root) if args.data_root else ROMAV2_DATA / "megadepth"
    model = flashim.load(args.model, preset=args.preset, engine=args.engine, **(json.loads(args.extra) if args.extra else {}))
    started = time.time()
    pool = SeededPosePool(args.workers, args.seed)
    pairs = mega1500_sparse(
        model, data_root, pool, args.max_pairs, {"workers": args.loaders, "depth": args.prefetch},
        {"resize": args.resize, "top_k": args.top_k}, args.bgr,
    )
    errors = pool.errors()
    result = {
        "dataset": "mega1500",
        "model": args.model,
        "preset": args.preset,
        "engine": args.engine,
        "resize": args.resize,
        "top_k": args.top_k,
        "bgr": args.bgr,
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
