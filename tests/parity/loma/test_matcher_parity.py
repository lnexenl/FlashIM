"""FlashIM's LoMa matcher vs upstream LoMa (vendored in ``tests/oracles/loma``) on identical keypoints and
descriptors.

The features come from the oracle's own ``detect_and_describe`` (path API, FP32 extractor); FlashIM gets them
as ``SparseFeatures`` in its pixel convention (``x = W (x_n + 1) / 2 - 0.5``) and converts back. Thresholds,
measured on sm_120 for every variant on both pairs:

- FP32 vs upstream ``mp=False``: identical match sets; max |P - P_ref| of the assignment product 9e-5 (bound
  1e-3).
- BF16 vs FP32 upstream: match-set IoU 0.908 to 0.998 (bound 0.88), above upstream's own ``mp=True`` vs
  ``mp=False`` (0.788 to 0.974: upstream's autocast also runs the positional projection in BF16, FlashIM keeps
  it in FP32).
- BF16 vs upstream ``mp=True``: IoU 0.774 to 0.974 (bound 0.7).
"""

import dataclasses
import sys

import pytest
import torch

from flashim.paths import ASSETS, MEGADEPTH_IMAGES, ORACLES

ORACLE = ORACLES / "loma"
VARIANTS = ("b128", "b", "l", "g", "r")
PAIRS = {
    "toronto": (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg"),
    "megadepth": (MEGADEPTH_IMAGES / "505127_40b48ea397_o.jpg", MEGADEPTH_IMAGES / "505128_561e05e146_o.jpg"),
}


@pytest.fixture(scope="module")
def upstream():
    sys.path.insert(0, str(ORACLE))
    try:
        import loma.loma
        import loma_oracle
    finally:
        sys.path.remove(str(ORACLE))
    return loma.loma, loma_oracle


def _pairs(m0: torch.Tensor) -> set[tuple[int, int]]:
    valid = m0[0] > -1
    return set(zip(torch.where(valid)[0].tolist(), m0[0][valid].tolist()))


def _iou(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


@pytest.fixture(scope="module", params=VARIANTS)
def reference(request, upstream):
    """Per variant and pair: the oracle's features (as SparseFeatures too) and match sets, ``mp`` off / on."""
    from flashim.models.loma import weights
    from flashim.models.weights_util import checkpoint_dir
    from flashim.types import SparseFeatures

    variant = request.param
    checkpoint = checkpoint_dir() / weights.CHECKPOINTS[variant].filename
    if not torch.cuda.is_available() or not checkpoint.exists():
        pytest.skip(f"needs CUDA and {checkpoint}")
    module, loma_oracle = upstream
    previous = torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        oracle = loma_oracle.load_oracle(variant, checkpoint, mp=False, extractor_amp=False)
        out = {}
        for name, images in PAIRS.items():
            feats, raw = [], []
            for image in images:
                k, d, h, w = oracle.detect_and_describe(str(image))
                pixels = torch.stack((w * (k[..., 0] + 1) / 2, h * (k[..., 1] + 1) / 2), -1) - 0.5
                feats.append(SparseFeatures(pixels, k.new_zeros(k.shape[:2]), d, (h, w)))
                raw.append((k, d))
            (ka, da), (kb, db) = raw
            result = {"features": feats, "raw": raw}
            for mp in (False, True):
                oracle.cfg = dataclasses.replace(oracle.cfg, mp=mp)
                with torch.inference_mode():
                    scores = oracle(ka, kb, da, db)["scores"]
                result[mp] = (scores, _pairs(module.filter_matches(scores, oracle.cfg.filter_threshold)[0]))
            out[name] = result
        del oracle
        torch.cuda.empty_cache()
        yield variant, out
    finally:
        torch.backends.cudnn.allow_tf32 = previous[0]
        torch.set_float32_matmul_precision(previous[1])


def _ours(model, result):
    a, b = result["features"]
    with torch.inference_mode():
        matches = model.match_features(a, b)
        enc = [model.encode(f.keypoints, f.image_size) for f in (a, b)]
        probs = model.scores(a.descriptors, b.descriptors, *enc).exp()
    return set(zip(matches.indices_A.tolist(), matches.indices_B.tolist())), probs


@pytest.mark.cuda
@pytest.mark.slow
def test_fp32_matches_upstream(reference, fp32_highest):
    import flashim

    variant, out = reference
    model = flashim.load("loma-matcher", "fp32", config={"variant": variant})
    for name, result in out.items():
        pairs, probs = _ours(model, result)
        ref_scores, ref_pairs = result[False]
        assert pairs == ref_pairs, (name, len(pairs), len(ref_pairs), len(pairs & ref_pairs))
        assert (probs - ref_scores).abs().max().item() < 1e-3, name


@pytest.mark.cuda
@pytest.mark.slow
def test_bf16_tracks_upstream(reference):
    import flashim

    variant, out = reference
    model = flashim.load("loma-matcher", "bf16", config={"variant": variant})
    assert model.compute_dtype == torch.bfloat16
    for name, result in out.items():
        pairs, _ = _ours(model, result)
        fp32, mixed = result[False][1], result[True][1]
        assert _iou(pairs, fp32) > 0.88, (name, _iou(pairs, fp32))
        assert _iou(pairs, fp32) >= _iou(mixed, fp32), (name, _iou(pairs, fp32), _iou(mixed, fp32))
        assert _iou(pairs, mixed) > 0.7, (name, _iou(pairs, mixed))


def test_random_weights_match_upstream_modules(upstream):
    """Without a checkpoint: upstream's matcher modules with random weights, converted by FlashIM's loader,
    give the same assignment (positions, input projection, Wqkv reordering, last-layer dual softmax)."""
    from flashim.models.lightglue.weights import convert_state_dict
    from flashim.models.loma import weights
    from flashim.models.loma.matcher import VARIANTS as CONFIGS
    from flashim.models.loma.matcher import LoMaMatcher

    module, _ = upstream
    config = dataclasses.replace(CONFIGS["b128"], input_dim=16, dim=32, num_heads=2, num_layers=2)
    torch.manual_seed(0)
    posenc = module.LearnableFourierPositionalEncoding(2, 16, 16, gamma=1.0)
    proj = torch.nn.Linear(16, 32)
    layers = torch.nn.ModuleList(module.TransformerLayer(32, 2) for _ in range(2))
    heads = torch.nn.ModuleList(module.MatchAssignment(32) for _ in range(2))
    for p in [*layers.parameters(), *heads.parameters()]:
        p.data.normal_(0, 0.3)
    state = {"input_proj.weight": proj.weight, "input_proj.bias": proj.bias,
             "posenc.Wr.weight": posenc.Wr.weight,
             **{f"transformers.{k}": v for k, v in layers.state_dict().items()},
             **{f"log_assignment.{k}": v for k, v in heads.state_dict().items()},
             "_detector.encoder.layers.0.weight": torch.zeros(1)}
    matcher = weights.split_state_dict(state, n_layers=2).matcher
    ours = LoMaMatcher.from_pretrained(
        config, state_dict=convert_state_dict(matcher, 2, 2), device="cpu"
    )
    kpts = [torch.rand(1, n, 2) * 2 - 1 for n in (7, 9)]
    desc = [torch.randn(1, n, 16) for n in (7, 9)]
    with torch.inference_mode():
        x0, x1 = proj(desc[0]), proj(desc[1])
        enc0, enc1 = posenc(kpts[0]), posenc(kpts[1])
        for layer in layers:
            x0, x1 = layer(x0, x1, enc0, enc1)
        ref = heads[1].eval()(x0, x1)[0]
        got = ours.scores(desc[0], desc[1], ours.posenc(kpts[0]), ours.posenc(kpts[1])).exp()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-6)
