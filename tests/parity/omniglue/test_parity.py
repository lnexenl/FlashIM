"""FlashIM OmniGlue parity tests against upstream google-research/omniglue (vendored in ``tests/oracles/omniglue``).

Covers:
- Pure-Python TF TensorBundle reader vs ``tf.train.load_checkpoint`` (bit-identical for all 72 ``sp_v6`` and 317 ``og_export`` tensors)
- Component parity: SuperPoint ``sp_v6`` (exact grayscale tensor & raw image), DINOv2 ViT-B/14 features + bilinear lookup,
  and ``OmniGlueMatcher`` on shared SuperPoint + DINOv2 inputs (CPU & CUDA)
- End-to-end parity: ``flashim.load("omniglue", "fp32")`` vs upstream ``OmniGlue`` on ``assets/sacre_coeur{1,2}.jpg``
  and ``assets/toronto_{A,B}.jpg``
- Reduced-precision tracking: ``bf16`` vs ``fp32``

The TF-dependent tests call ``tests/oracles/omniglue/tf_oracle.py`` in a separate TensorFlow-CPU venv
(``$FLASHIM_OMNIGLUE_TF_PYTHON`` or ``~/work/oracles/omniglue/.venv/bin/python``) and skip cleanly when absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

import flashim
from flashim.models.omniglue import weights as og_weights
from flashim.models.omniglue.extractor import dino_size, sample_dino_descriptors
from flashim.models.omniglue.tf_bundle import read_bundle_from_archive
from flashim.models.superpoint import weights as sp_weights
from flashim.models.weights_util import find_weights

ASSETS = Path(__file__).resolve().parents[3] / "assets"
ORACLE_DIR = Path(__file__).resolve().parents[2] / "oracles" / "omniglue"
TF_ORACLE = ORACLE_DIR / "tf_oracle.py"

pytestmark = [pytest.mark.oracle, pytest.mark.weights]


def _tf_python() -> Path | None:
    env = os.environ.get("FLASHIM_OMNIGLUE_TF_PYTHON")
    candidates = [Path(env)] if env else [Path.home() / "work" / "oracles" / "omniglue" / ".venv" / "bin" / "python"]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _require_tf_python() -> Path:
    py = _tf_python()
    if py is None:
        pytest.skip("TF-CPU oracle venv not found (set FLASHIM_OMNIGLUE_TF_PYTHON)")
    return py


def _run_tf_oracle(mode: str, *, sp_path: Path | None = None, og_path: Path | None = None,
                   inputs: dict[str, np.ndarray] | None = None) -> dict:
    py = _require_tf_python()
    with tempfile.TemporaryDirectory(prefix="flashim_og_parity_") as tmp:
        tmp_dir = Path(tmp)
        cmd = [str(py), str(TF_ORACLE), "--mode", mode]
        if sp_path is not None:
            cmd += ["--sp-export", str(sp_path)]
        if og_path is not None:
            cmd += ["--og-export", str(og_path)]
        if inputs is not None:
            in_file = tmp_dir / "in.npz"
            np.savez(in_file, **inputs)
            cmd += ["--input", str(in_file)]
        out_file = tmp_dir / ("out.json" if mode == "bundle_hashes" else "out.npz")
        cmd += ["--output", str(out_file)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"tf_oracle ({mode}) failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        if mode == "bundle_hashes":
            return json.loads(out_file.read_text())
        with np.load(out_file) as data:
            return {k: data[k] for k in data.files}


def _load_upstream_dino(dino_ckpt: Path, device: str):
    """Load vendored upstream ``third_party.dinov2.dino.vit_base()``."""
    oracle_str = str(ORACLE_DIR)
    sys.path.insert(0, oracle_str)
    try:
        from third_party.dinov2 import dino as upstream_dino
    finally:
        if oracle_str in sys.path:
            sys.path.remove(oracle_str)
    model = upstream_dino.vit_base()
    model.load_state_dict(torch.load(dino_ckpt, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def _upstream_lookup_dino(dino_hwc: np.ndarray, kpts_xy: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Pure-numpy mirror of upstream ``dino_extract.get_dino_descriptors`` + ``utils.lookup_descriptor_bilinear``."""
    dh, dw = dino_size(orig_h, orig_w, 630, 14)
    hf, wf = dh // 14, dw // 14
    kp_feat = kpts_xy.astype(np.float32) / np.array([orig_w, orig_h], dtype=np.float32) * np.array([wf, hf], dtype=np.float32)
    out = np.zeros((len(kp_feat), dino_hwc.shape[-1]), dtype=np.float32)
    for idx, (x, y) in enumerate(kp_feat):
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        xs = [x0] + ([x0 + 1] if (not float(x).is_integer() and x < wf - 1) else [])
        ys = [y0] + ([y0 + 1] if (not float(y).is_integer() and y < hf - 1) else [])
        acc = np.zeros(dino_hwc.shape[-1], dtype=np.float64)
        for cx in xs:
            for cy in ys:
                acc += (1.0 - abs(float(x) - cx)) * (1.0 - abs(float(y) - cy)) * dino_hwc[cy, cx]
        out[idx] = acc.astype(np.float32)
    return out


def _upstream_dino_features(upstream_vit, img_rgb: np.ndarray, device: str) -> np.ndarray:
    """Upstream ``DINOExtract.forward`` (cv2.INTER_LINEAR resize -> ImageNet norm -> vit_base)."""
    h, w = img_rgb.shape[:2]
    dh, dw = dino_size(h, w, 630, 14)
    resized = cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy((resized / 255.0 - mean) / std).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        feat = upstream_vit.get_intermediate_layers(x, n=1)[0]
        feat = feat.reshape(1, dh // 14, dw // 14, -1)[0].cpu().numpy()
    return feat


def _match_agreement(ref_a: np.ndarray, ref_b: np.ndarray, got_a: np.ndarray, got_b: np.ndarray, radius: float = 2.0) -> float:
    if len(ref_a) == 0:
        return 1.0 if len(got_a) == 0 else 0.0
    if len(got_a) == 0:
        return 0.0
    da = torch.cdist(torch.from_numpy(ref_a).float(), torch.from_numpy(got_a).float(), compute_mode="donot_use_mm_for_euclid_dist")
    db = torch.cdist(torch.from_numpy(ref_b).float(), torch.from_numpy(got_b).float(), compute_mode="donot_use_mm_for_euclid_dist")
    return float(((da < radius) & (db < radius)).any(1).float().mean())


def test_tf_bundle_reader_bit_identical():
    """Pure-Python TensorBundle v2 reader matches ``tf.train.load_checkpoint`` bit-for-bit on all tensors."""
    sp_arc = find_weights(sp_weights.CHECKPOINTS["sp_v6"], env=sp_weights.CHECKPOINT_SP_V6_ENV)
    og_arc = find_weights(og_weights.CHECKPOINTS["omniglue"], env=og_weights.CHECKPOINT_ENV)
    ref = _run_tf_oracle("bundle_hashes", sp_path=sp_arc, og_path=og_arc)
    for label, arc in (("sp_v6", sp_arc), ("og_export", og_arc)):
        got = read_bundle_from_archive(arc)
        assert set(got) == set(ref[label])
        for name, tensor in got.items():
            arr = np.ascontiguousarray(tensor.numpy())
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            assert [list(arr.shape), str(arr.dtype), digest] == ref[label][name], f"{label}:{name}"


def test_dinov2_vitb14_component_parity():
    """FlashIM DINOv2 ViT-B/14 feature map and bilinear sampling match upstream ``third_party/dinov2``."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_ckpt = find_weights(og_weights.CHECKPOINTS["dinov2"], env=og_weights.DINOV2_ENV)
    upstream_vit = _load_upstream_dino(dino_ckpt, device)
    extractor = flashim.load("omniglue-extractor", "fp32", device=device)

    img = np.array(Image.open(ASSETS / "sacre_coeur1.jpg").convert("RGB"))
    h, w = img.shape[:2]
    dh, dw = dino_size(h, w, 630, 14)
    resized = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy((resized / 255.0 - mean) / std).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        prev_tf32 = torch.backends.cudnn.allow_tf32
        prev_mm_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            ref_tokens = upstream_vit.get_intermediate_layers(x, n=1)[0]
            ref_map = ref_tokens.reshape(1, dh // 14, dw // 14, -1).permute(0, 3, 1, 2)
            got_map = extractor.dinov2(x)
        finally:
            torch.backends.cudnn.allow_tf32 = prev_tf32
            torch.backends.cuda.matmul.allow_tf32 = prev_mm_tf32

    max_diff = float((got_map - ref_map).abs().max())
    mean_diff = float((got_map - ref_map).abs().mean())
    min_cos = float(torch.nn.functional.cosine_similarity(got_map, ref_map, dim=1).min())
    assert max_diff < 5e-3, f"DINOv2 ViT-B/14 map max diff {max_diff:.2e}"
    assert mean_diff < 1e-4, f"DINOv2 ViT-B/14 map mean diff {mean_diff:.2e}"
    assert min_cos > 0.99999, f"DINOv2 ViT-B/14 min cosine {min_cos:.6f}"

    # Verify bilinear descriptor lookup including boundary points near [W-1, W) and [H-1, H)
    rng = np.random.default_rng(7)
    kpts = np.stack([rng.uniform(0, w - 0.1, 256), rng.uniform(0, h - 0.1, 256)], axis=-1).astype(np.float32)
    kpts[:4] = [[0.0, 0.0], [w - 0.5, h - 0.5], [w - 1.0, 0.0], [0.0, h - 1.0]]
    ref_desc = _upstream_lookup_dino(ref_map[0].permute(1, 2, 0).cpu().numpy(), kpts, h, w)
    got_desc = sample_dino_descriptors(ref_map, torch.from_numpy(kpts).to(device), h, w).cpu().numpy()
    assert np.abs(got_desc - ref_desc).max() < 1e-4


def test_sp_v6_and_matcher_component_parity():
    """Component parity vs TF SavedModels: SuperPoint ``sp_v6`` on exact gray input + ``OmniGlueMatcher`` on shared inputs."""
    sp_arc = find_weights(sp_weights.CHECKPOINTS["sp_v6"], env=sp_weights.CHECKPOINT_SP_V6_ENV)
    og_arc = find_weights(og_weights.CHECKPOINTS["omniglue"], env=og_weights.CHECKPOINT_ENV)
    img0 = np.array(Image.open(ASSETS / "sacre_coeur1.jpg").convert("RGB"))
    img1 = np.array(Image.open(ASSETS / "sacre_coeur2.jpg").convert("RGB"))

    sp_ref = _run_tf_oracle("sp_v6", sp_path=sp_arc, inputs={"img0": img0, "img1": img1, "top_k": np.int32(1024)})

    # 1. SuperPoint sp_v6 parity on the exact preprocessed grayscale tensor
    sp_model = flashim.load("superpoint", "fp32", device="cpu", overrides={"variant": "sp_v6"})
    g0 = torch.from_numpy(sp_ref["g0"]).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        kpts, scores, desc = sp_model.detect(g0, 1024)

    prob0 = sp_ref["prob0"][0]
    desc0_dense = sp_ref["desc0_dense"][0]
    tf_ys, tf_xs = np.where(prob0 > 0)
    tf_sc = prob0[tf_ys, tf_xs]
    order = np.argsort(tf_sc)[-1024:]
    tf_kpts = np.stack([tf_xs[order], tf_ys[order]], -1).astype(np.float32)
    tf_sc = tf_sc[order]
    tf_desc = np.stack([desc0_dense[int(y), int(x)] for x, y in tf_kpts], axis=0)

    mine = {tuple(k): (s, d) for k, s, d in zip(kpts.tolist(), scores.tolist(), desc.numpy())}
    theirs = {tuple(k): (s, d) for k, s, d in zip(tf_kpts.tolist(), tf_sc.tolist(), tf_desc)}
    common = mine.keys() & theirs.keys()
    assert len(common) >= 0.995 * len(theirs), f"sp_v6 common {len(common)}/{len(theirs)}"
    assert max(abs(mine[k][0] - theirs[k][0]) for k in common) < 5e-5
    assert min(float(mine[k][1] @ theirs[k][1]) for k in common) > 0.99999

    # 2. OmniGlue matcher parity on shared SP + real DINOv2 inputs
    dino_ckpt = find_weights(og_weights.CHECKPOINTS["dinov2"], env=og_weights.DINOV2_ENV)
    upstream_vit = _load_upstream_dino(dino_ckpt, "cpu")
    dino_map0 = _upstream_dino_features(upstream_vit, img0, "cpu")
    dino_map1 = _upstream_dino_features(upstream_vit, img1, "cpu")
    dino0 = _upstream_lookup_dino(dino_map0, sp_ref["kp0"], *img0.shape[:2])
    dino1 = _upstream_lookup_dino(dino_map1, sp_ref["kp1"], *img1.shape[:2])

    og_ref = _run_tf_oracle(
        "matcher", og_path=og_arc,
        inputs={
            "kp0": sp_ref["kp0"], "kp1": sp_ref["kp1"],
            "desc0": sp_ref["desc0"], "desc1": sp_ref["desc1"],
            "score0": sp_ref["score0"], "score1": sp_ref["score1"],
            "dino0": dino0, "dino1": dino1,
            "hw0": np.array(img0.shape[:2], dtype=np.int32),
            "hw1": np.array(img1.shape[:2], dtype=np.int32),
            "threshold": np.float32(1e-3),
        },
    )

    matcher = flashim.load("omniglue-matcher", "fp32", device="cpu")
    with torch.inference_mode():
        sa = matcher.soft_assignment(
            torch.from_numpy(sp_ref["kp0"])[None], torch.from_numpy(sp_ref["kp1"])[None],
            torch.from_numpy(sp_ref["desc0"])[None], torch.from_numpy(sp_ref["desc1"])[None],
            torch.from_numpy(sp_ref["score0"])[None], torch.from_numpy(sp_ref["score1"])[None],
            torch.from_numpy(dino0)[None], torch.from_numpy(dino1)[None],
            img0.shape[:2], img1.shape[:2],
        ).numpy()
    max_diff = float(np.abs(sa - og_ref["soft_assignment"]).max())
    mean_diff = float(np.abs(sa - og_ref["soft_assignment"]).mean())
    assert max_diff < 5e-4, f"OmniGlueMatcher soft_assignment max diff {max_diff:.2e}"
    assert mean_diff < 1e-6, f"OmniGlueMatcher soft_assignment mean diff {mean_diff:.2e}"
    # Verify exact match set agreement on identical inputs
    from flashim.models.omniglue.model import OmniGlueFeatures

    fa = OmniGlueFeatures(
        torch.from_numpy(sp_ref["kp0"])[None], torch.from_numpy(sp_ref["score0"])[None],
        torch.from_numpy(sp_ref["desc0"])[None], img0.shape[:2], torch.from_numpy(dino0)[None],
    )
    fb = OmniGlueFeatures(
        torch.from_numpy(sp_ref["kp1"])[None], torch.from_numpy(sp_ref["score1"])[None],
        torch.from_numpy(sp_ref["desc1"])[None], img1.shape[:2], torch.from_numpy(dino1)[None],
    )
    m = matcher.match_features(fa, fb, threshold=1e-3)
    agree_shared = _match_agreement(
        og_ref["match_kp0"], og_ref["match_kp1"], m.keypoints_A.numpy(), m.keypoints_B.numpy(), radius=1e-3
    )
    assert agree_shared > 0.99, f"Shared-input matcher agreement {agree_shared:.4f}"


@pytest.mark.cuda
@pytest.mark.parametrize(
    "pair",
    [
        ("sacre_coeur1.jpg", "sacre_coeur2.jpg"),
        ("megadepth_0015/505127_40b48ea397_o.jpg", "megadepth_0015/505128_561e05e146_o.jpg"),
    ],
)
def test_end_to_end_pipeline_matches_upstream(pair):
    """End-to-end ``flashim.load('omniglue', 'fp32')`` matches upstream ``OmniGlue.FindMatches`` on asset pairs."""
    sp_arc = find_weights(sp_weights.CHECKPOINTS["sp_v6"], env=sp_weights.CHECKPOINT_SP_V6_ENV)
    og_arc = find_weights(og_weights.CHECKPOINTS["omniglue"], env=og_weights.CHECKPOINT_ENV)
    dino_ckpt = find_weights(og_weights.CHECKPOINTS["dinov2"], env=og_weights.DINOV2_ENV)

    p0, p1 = ASSETS / pair[0], ASSETS / pair[1]
    img0 = np.array(Image.open(p0).convert("RGB"))
    img1 = np.array(Image.open(p1).convert("RGB"))

    # Upstream end-to-end (sp_v6 TF + DINOv2 PyTorch + og_export TF)
    sp_ref = _run_tf_oracle("sp_v6", sp_path=sp_arc, inputs={"img0": img0, "img1": img1, "top_k": np.int32(1024)})
    upstream_vit = _load_upstream_dino(dino_ckpt, "cuda")
    dino0 = _upstream_lookup_dino(_upstream_dino_features(upstream_vit, img0, "cuda"), sp_ref["kp0"], *img0.shape[:2])
    dino1 = _upstream_lookup_dino(_upstream_dino_features(upstream_vit, img1, "cuda"), sp_ref["kp1"], *img1.shape[:2])
    og_ref = _run_tf_oracle(
        "matcher", og_path=og_arc,
        inputs={
            "kp0": sp_ref["kp0"], "kp1": sp_ref["kp1"],
            "desc0": sp_ref["desc0"], "desc1": sp_ref["desc1"],
            "score0": sp_ref["score0"], "score1": sp_ref["score1"],
            "dino0": dino0, "dino1": dino1,
            "hw0": np.array(img0.shape[:2], dtype=np.int32),
            "hw1": np.array(img1.shape[:2], dtype=np.int32),
            "threshold": np.float32(0.02),
        },
    )

    # FlashIM end-to-end (CUDA fp32)
    prev_tf32 = torch.backends.cudnn.allow_tf32
    prev_mm_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model = flashim.load("omniglue", "fp32", device="cuda")
        matches = model.match(img0, img1, max_matches=1024, threshold=0.02)
    finally:
        torch.backends.cudnn.allow_tf32 = prev_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_mm_tf32

    agree = _match_agreement(
        og_ref["match_kp0"], og_ref["match_kp1"],
        matches.keypoints_A.cpu().numpy(), matches.keypoints_B.cpu().numpy(),
        radius=2.0,
    )
    assert len(og_ref["match_kp0"]) > 50
    # Slight keypoint top-k boundary shifts occur when upstream uint8 cv2.resize/cvtColor differs by <= 0.5/255
    # from float32 bilinear resize + RGB->gray (component parity above verifies exact math to <5e-4 / >99%).
    assert agree > 0.85, f"{pair}: end-to-end match agreement {agree:.4f} (ref={len(og_ref['match_kp0'])}, got={len(matches)})"


@pytest.mark.cuda
def test_bf16_tracks_fp32():
    """``bf16`` preset tracks ``fp32`` closely on ``assets/sacre_coeur{1,2}.jpg``."""
    img0 = np.array(Image.open(ASSETS / "sacre_coeur1.jpg").convert("RGB"))
    img1 = np.array(Image.open(ASSETS / "sacre_coeur2.jpg").convert("RGB"))
    m_fp32 = flashim.load("omniglue", "fp32", device="cuda").match(img0, img1, threshold=0.02)
    m_bf16 = flashim.load("omniglue", "bf16", device="cuda").match(img0, img1, threshold=0.02)
    agree = _match_agreement(
        m_fp32.keypoints_A.cpu().numpy(), m_fp32.keypoints_B.cpu().numpy(),
        m_bf16.keypoints_A.cpu().numpy(), m_bf16.keypoints_B.cpu().numpy(),
        radius=2.0,
    )
    assert agree > 0.85, f"bf16 vs fp32 agreement {agree:.4f}"
