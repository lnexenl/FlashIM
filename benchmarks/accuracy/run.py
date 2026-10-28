"""Relative-pose accuracy of a FlashIM dense matcher on MegaDepth-1500 / ScanNet-1500.

Protocol (identical to RoMaV2's benchmarks): per pair one ``match``; MegaDepth
draws 5 fresh ``sample(5000)`` sets per pair (intrinsics rescaled to a 1200px long
side), ScanNet samples once and reshuffles 5 times (480px short side, -0.5 px
offset); OpenCV essential-matrix RANSAC with ``0.5px`` normalized threshold;
AUC@5/10/20 of ``max(e_t, e_R)``.

    python benchmarks/accuracy/run.py --dataset mega1500 --preset fp8 --engine compile --seed 17
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))  # before numpy/torch/cv2 load their OpenMP runtimes

import argparse
import json
import logging
import os.path as osp
import random
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
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
from flashim.paths import DATA

logger = logging.getLogger(__name__)
MEGA_SCENES = ("0015_0.1_0.3.npz", "0015_0.3_0.5.npz", "0022_0.1_0.3.npz", "0022_0.3_0.5.npz", "0022_0.5_0.7.npz")


def _pose_error(seed: int, kpts1, kpts2, K1, K2, R, t) -> float:
    cv2.setRNGSeed(seed)  # OpenCV's RNG is thread-local; seed per job so results ignore thread scheduling
    norm_threshold = 0.5 / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
    try:
        R_est, t_est, _ = estimate_pose_cv2_ransac(kpts1, kpts2, K1, K2, norm_threshold, conf=0.99999)
        e_t, e_R = compute_pose_error(R_est, t_est[:, 0], R, t)
        return max(e_t, e_R)
    except Exception as exc:  # degenerate geometry counts as a failure (90 degrees)
        logger.debug("pose estimation failed: %s", exc)
        return 90.0


class PosePool:
    """Consumer side of the pipeline: RANSAC jobs run on worker threads (cv2 releases the GIL)
    while the caller thread keeps the GPU busy with match/sample.

    Submission blocks once ``max_pending`` jobs are in flight so the GPU cannot run unboundedly
    ahead. Errors are returned in submission order, so results do not depend on thread timing.
    """

    def __init__(self, workers: int, max_pending: int = 0):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pose")
        self._slots = threading.Semaphore(max_pending or 4 * workers)
        self._futures: list[Future] = []

    def submit(self, *job) -> None:
        self._slots.acquire()
        fut = self._pool.submit(_pose_error, len(self._futures), *job)
        fut.add_done_callback(lambda _: self._slots.release())
        self._futures.append(fut)

    def errors(self) -> list[float]:
        try:
            return [f.result() for f in self._futures]
        finally:
            self._pool.shutdown()


def prefetch(items, load, workers: int, depth: int):
    """Yield ``load(item)`` in order, decoding up to ``depth`` items ahead on ``workers`` threads
    (PIL decode releases the GIL), so the GPU never waits on disk/JPEG."""
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="load") as pool:
        pending: deque[Future] = deque()
        it = iter(items)
        for item in it:
            pending.append(pool.submit(load, item))
            if len(pending) >= depth:
                break
        while pending:
            result = pending.popleft().result()
            for item in it:
                pending.append(pool.submit(load, item))
                break
            yield result


def _read_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def mega1500(model, data_root: Path, pool: PosePool, max_pairs: int, loader: dict) -> int:
    def jobs():
        for scene_name in MEGA_SCENES:
            scene = np.load(data_root / scene_name, allow_pickle=True)
            pairs, intrinsics, poses, paths = (scene[k] for k in ("pair_infos", "intrinsics", "poses", "image_paths"))
            n = len(pairs) if max_pairs <= 0 else min(len(pairs), max(1, max_pairs // 5))
            for i in range(n):
                idx1, idx2 = pairs[i][0]
                T1, T2 = poses[idx1], poses[idx2]
                R, t = compute_relative_pose(T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3])
                yield (f"{data_root}/{paths[idx1]}", f"{data_root}/{paths[idx2]}",
                       intrinsics[idx1].copy(), intrinsics[idx2].copy(), R, t)

    def load(job):
        return (*job, _read_rgb(job[0]), _read_rgb(job[1]))

    count = 0
    for *_, K1, K2, R, t, img_a, img_b in tqdm(prefetch(jobs(), load, **loader), desc="mega1500", leave=False):
        preds = model.match_dense(img_a, img_b)
        (h1, w1), (h2, w2) = img_a.shape[:2], img_b.shape[:2]
        s1, s2 = 1200 / max(w1, h1), 1200 / max(w2, h2)
        w1, h1, w2, h2 = s1 * w1, s1 * h1, s2 * w2, s2 * h2
        K1[:2] *= s1
        K2[:2] *= s2
        for _ in range(5):
            matches, *_ = model.sample(preds, 5000)
            k1, k2 = model.to_pixel_coordinates(matches, h1, w1, h2, w2)
            k1, k2 = k1.cpu().numpy(), k2.cpu().numpy()
            perm = np.random.permutation(len(k1))
            pool.submit(k1[perm], k2[perm], K1, K2, R, t)
        count += 1
    return count


def scannet1500(model, data_root: Path, pool: PosePool, max_pairs: int, loader: dict) -> int:
    data = np.load(osp.join(data_root, "test.npz"))
    pairs, rel_pose = data["name"], data["rel_pose"]
    if max_pairs > 0:
        pairs, rel_pose = pairs[:max_pairs], rel_pose[:max_pairs]

    def jobs():
        for scene, pose in zip(pairs, rel_pose):
            scene_dir = osp.join(data_root, "scans_test", f"scene0{scene[0]}_00")
            T = pose.reshape(3, 4)
            with open(osp.join(scene_dir, "intrinsic", "intrinsic_color.txt")) as handle:
                K = np.stack([np.array([float(v) for v in row.split()]) for row in handle.read().split("\n") if row])
            yield (osp.join(scene_dir, "color", f"{scene[2]}.jpg"), osp.join(scene_dir, "color", f"{scene[3]}.jpg"),
                   K, T[:3, :3], T[:3, 3])

    def load(job):
        return (*job, _read_rgb(job[0]), _read_rgb(job[1]))

    for *_, K, R, t, img_a, img_b in tqdm(prefetch(jobs(), load, **loader), total=len(pairs), desc="scannet1500", leave=False):
        (h1, w1), (h2, w2) = img_a.shape[:2], img_b.shape[:2]
        preds = model.match_dense(img_a, img_b)
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
            k1, k2 = k1[perm], k2[perm]  # fancy indexing copies, so queued jobs keep their own arrays
            pool.submit(k1, k2, K1, K2, R, t)
    return len(pairs)


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
    parser.add_argument("--attention", default="sdpa", choices=("sdpa", "sage", "sage-fp8"))
    parser.add_argument("--engine", default="compile", choices=("eager", "compile", "compile-full"))
    parser.add_argument("--data-root", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all 1500")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4), help="RANSAC threads")
    parser.add_argument("--loaders", type=int, default=4, help="image decode threads")
    parser.add_argument("--prefetch", type=int, default=16, help="pairs decoded ahead of the GPU")
    parser.add_argument("--cv-threads", type=int, default=0, help="OpenCV/OMP/torch CPU threads (0 = all cores)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    ncpu = args.cv_threads or os.cpu_count() or 4
    cv2.setNumThreads(ncpu)
    torch.set_num_threads(ncpu)  # OMP_NUM_THREADS must be set before launch to affect other OpenMP libs

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    data_root = Path(args.data_root) if args.data_root else (
        DATA / "megadepth" if args.dataset == "mega1500" else DATA / "scannet" / "scans"
    )
    model = flashim.load(
        args.model, args.preset, engine=args.engine,
        overrides={"arch.setting": args.dataset, "optimize.attention": args.attention},
    )
    started = time.time()
    run = mega1500 if args.dataset == "mega1500" else scannet1500
    pool = PosePool(args.workers)
    pairs = run(model, data_root, pool, args.max_pairs, dict(workers=args.loaders, depth=args.prefetch))
    errors = pool.errors()  # drains the queue; GPU stage is already finished
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "preset": args.preset,
        "attention": args.attention,
        "engine": args.engine,
        "seed": args.seed,
        "workers": args.workers,
        "loaders": args.loaders,
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
