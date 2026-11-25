"""Subprocess oracle runner executed in a separate TensorFlow-CPU virtual environment.

Not upstream code (FlashIM test helper). Run with the TF-CPU venv (``$FLASHIM_OMNIGLUE_TF_PYTHON`` or
``~/work/oracles/omniglue/.venv/bin/python``); exchanges arrays via ``.npz`` files in a temporary directory
so FlashIM's main venv stays free of TensorFlow.

Modes:
- ``sp_v6``: runs upstream ``SuperPointExtract`` on ``img0`` / ``img1`` and also captures raw ``prob_nms`` /
  ``descriptors`` on the preprocessed grayscale ``g0``.
- ``matcher``: runs upstream ``og_export`` ``serving_default`` + ``soft_assignment_to_match_matrix`` on shared
  SuperPoint + DINOv2 inputs.
- ``bundle_hashes``: dumps SHA-256 of every numeric tensor from ``tf.train.load_checkpoint`` for both SavedModels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent
if str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))


def _ensure_dir(path: Path, tmp_root: Path) -> Path:
    if path.is_dir():
        return path
    suffixes = "".join(path.suffixes).lower()
    out = tmp_root / path.stem
    out.mkdir(parents=True, exist_ok=True)
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out)
    elif suffixes.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(path, "r:*") as tf_arc:
            tf_arc.extractall(out)
    else:
        raise ValueError(f"Unsupported SavedModel path: {path}")
    pb = next(out.rglob("saved_model.pb"))
    return pb.parent


def _run_sp_v6(sp_dir: Path, inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from omniglue import superpoint_extract

    sp = superpoint_extract.SuperPointExtract(str(sp_dir))
    img0, img1 = inp["img0"], inp["img1"]
    top_k = int(inp["top_k"]) if "top_k" in inp else 1024

    im0_r, _ = sp._resize_input_image(img0)
    g0 = sp._preprocess_image(im0_r)
    in_t = sp._graph.get_tensor_by_name("superpoint/image:0")
    p_t = sp._graph.get_tensor_by_name("superpoint/prob_nms:0")
    d_t = sp._graph.get_tensor_by_name("superpoint/descriptors:0")
    prob0, desc0_dense = sp._sess.run([p_t, d_t], feed_dict={in_t: g0[None]})

    kp0, desc0, score0 = sp(img0, num_features=top_k)
    kp1, desc1, score1 = sp(img1, num_features=top_k)
    return {
        "g0": g0,
        "prob0": prob0,
        "desc0_dense": desc0_dense,
        "kp0": kp0.astype(np.float32),
        "desc0": desc0.astype(np.float32),
        "score0": score0.astype(np.float32),
        "kp1": kp1.astype(np.float32),
        "desc1": desc1.astype(np.float32),
        "score1": score1.astype(np.float32),
    }


def _run_matcher(og_dir: Path, inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import tensorflow as tf
    from omniglue import utils

    matcher = tf.saved_model.load(str(og_dir)).signatures["serving_default"]
    h0, w0 = int(inp["hw0"][0]), int(inp["hw0"][1])
    h1, w1 = int(inp["hw1"][0]), int(inp["hw1"][1])
    threshold = float(inp["threshold"]) if "threshold" in inp else 1e-3

    og_in = {
        "keypoints0": tf.convert_to_tensor(inp["kp0"][None], dtype=tf.float32),
        "keypoints1": tf.convert_to_tensor(inp["kp1"][None], dtype=tf.float32),
        "descriptors0": tf.convert_to_tensor(inp["desc0"][None], dtype=tf.float32),
        "descriptors1": tf.convert_to_tensor(inp["desc1"][None], dtype=tf.float32),
        "scores0": tf.convert_to_tensor(inp["score0"][None, :, None], dtype=tf.float32),
        "scores1": tf.convert_to_tensor(inp["score1"][None, :, None], dtype=tf.float32),
        "descriptors0_dino": tf.convert_to_tensor(inp["dino0"][None], dtype=tf.float32),
        "descriptors1_dino": tf.convert_to_tensor(inp["dino1"][None], dtype=tf.float32),
        "width0": tf.convert_to_tensor([w0], dtype=tf.int32),
        "width1": tf.convert_to_tensor([w1], dtype=tf.int32),
        "height0": tf.convert_to_tensor([h0], dtype=tf.int32),
        "height1": tf.convert_to_tensor([h1], dtype=tf.int32),
    }
    soft_full = matcher(**og_in)["soft_assignment"]
    soft = soft_full[:, :-1, :-1]
    match_matrix = utils.soft_assignment_to_match_matrix(soft, threshold).numpy().squeeze(0)
    match_indices = np.argwhere(match_matrix)
    keep = [
        i for i in range(match_indices.shape[0])
        if inp["score0"][match_indices[i, 0]] > 0.0 and inp["score1"][match_indices[i, 1]] > 0.0
    ]
    match_indices = match_indices[keep] if keep else np.zeros((0, 2), dtype=np.int64)
    soft_np = soft.numpy()
    match_kp0 = inp["kp0"][match_indices[:, 0]] if len(match_indices) else np.zeros((0, 2), dtype=np.float32)
    match_kp1 = inp["kp1"][match_indices[:, 1]] if len(match_indices) else np.zeros((0, 2), dtype=np.float32)
    match_conf = (
        soft_np[0, match_indices[:, 0], match_indices[:, 1]]
        if len(match_indices) else np.zeros((0,), dtype=np.float32)
    )
    return {
        "soft_assignment": soft_full.numpy(),
        "match_kp0": match_kp0.astype(np.float32),
        "match_kp1": match_kp1.astype(np.float32),
        "match_conf": match_conf.astype(np.float32),
        "match_indices": match_indices.astype(np.int64),
    }


def _run_bundle_hashes(sp_dir: Path, og_dir: Path, out_json: Path) -> None:
    import tensorflow as tf

    result = {}
    for label, d in (("sp_v6", sp_dir), ("og_export", og_dir)):
        ckpt = str(d / "variables" / "variables")
        reader = tf.train.load_checkpoint(ckpt)
        tensors = {}
        for name, _ in tf.train.list_variables(ckpt):
            if name == "_CHECKPOINTABLE_OBJECT_GRAPH":
                continue
            arr = np.ascontiguousarray(reader.get_tensor(name))
            tensors[name] = [list(arr.shape), str(arr.dtype), hashlib.sha256(arr.tobytes()).hexdigest()]
        result[label] = tensors
    out_json.write_text(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("sp_v6", "matcher", "bundle_hashes"))
    parser.add_argument("--sp-export", type=Path)
    parser.add_argument("--og-export", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="omniglue_tf_oracle_") as tmp:
        tmp_path = Path(tmp)
        if args.mode == "bundle_hashes":
            sp_dir = _ensure_dir(args.sp_export, tmp_path)
            og_dir = _ensure_dir(args.og_export, tmp_path)
            _run_bundle_hashes(sp_dir, og_dir, args.output)
            return
        inp = dict(np.load(args.input))
        if args.mode == "sp_v6":
            out = _run_sp_v6(_ensure_dir(args.sp_export, tmp_path), inp)
        else:
            out = _run_matcher(_ensure_dir(args.og_export, tmp_path), inp)
        np.savez(args.output, **out)


if __name__ == "__main__":
    main()
