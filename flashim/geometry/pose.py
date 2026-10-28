"""Relative-pose estimation and pose-accuracy metrics (DKM/RoMa evaluation protocol)."""

from __future__ import annotations

import numpy as np

from flashim.utils import import_cv2


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ R1.T
    return rots, -rots @ t1 + t2


def angle_error_mat(R1, R2) -> float:
    cos = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)
    return float(np.rad2deg(np.abs(np.arccos(cos))))


def angle_error_vec(v1, v2) -> float:
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def compute_pose_error(R_gt, t_gt, R, t) -> tuple[float, float]:
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = min(error_t, 180 - error_t)  # essential-matrix sign ambiguity
    return error_t, angle_error_mat(R, R_gt)


def pose_auc(errors, thresholds) -> list[float]:
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1]]
        e = np.r_[errors[:last], t]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return aucs


def estimate_pose_cv2_ransac(kpts0, kpts1, K0, K1, norm_thresh: float, conf: float = 0.99999):
    """Essential matrix (OpenCV RANSAC) on intrinsics-normalized points, best candidate by inliers."""
    cv2 = import_cv2()

    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])
    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf)
    best = None
    if E is not None:
        best_inliers = 0
        for candidate in np.split(E, len(E) // 3):
            n, R, t, _ = cv2.recoverPose(candidate, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_inliers:
                best_inliers = n
                best = (R, t, mask.ravel() > 0)
    if best is None:
        raise RuntimeError("pose estimation failed")
    return best


def summarize(errors) -> dict[str, float]:
    errors = np.asarray(errors)
    auc = pose_auc(errors, [5, 10, 20])
    acc = {t: float((errors < t).mean()) for t in (5, 10, 15, 20)}
    return {
        "auc_5": auc[0],
        "auc_10": auc[1],
        "auc_20": auc[2],
        "map_5": acc[5],
        "map_10": float(np.mean([acc[5], acc[10]])),
        "map_20": float(np.mean([acc[5], acc[10], acc[15], acc[20]])),
    }
