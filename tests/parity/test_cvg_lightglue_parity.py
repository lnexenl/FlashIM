"""FlashIM LightGlue vs cvg/LightGlue (vendored in ``tests/oracles/lightglue``) for the per-extractor releases, on the same
features: FP32 match lists must be identical. Upstream is built without ``features=`` (that path downloads under a
versioned file name) and loads the checkpoint FlashIM resolves, through upstream's own key renaming."""

import sys

import pytest
import torch

from ..conftest import MEGADEPTH_IMAGES, ORACLES

pytestmark = [pytest.mark.cuda, pytest.mark.slow]
ORACLE = ORACLES / "lightglue"
IMAGES = [MEGADEPTH_IMAGES / name  # an overlapping pair: hundreds of matches
          for name in ("101776896_21fc24b031_o.jpg", "101795964_72e8f63ab1_o.jpg")]
# variant -> (extractor whose features it matches, descriptor dim)
VARIANTS = {"aliked": ("aliked", 128), "raco-aliked": ("raco-aliked", 128), "superpoint": ("superpoint", 256),
            "disk": ("disk", 128)}


@pytest.fixture(scope="module")
def upstream():
    if not (ORACLE / "lightglue").is_dir() or not all(path.exists() for path in IMAGES):
        pytest.skip("needs the MegaDepth images")
    sys.path.insert(0, str(ORACLE))
    try:
        from lightglue.lightglue import LightGlue
    finally:
        sys.path.remove(str(ORACLE))
    return LightGlue


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_fp32_matches_cvg_lightglue(upstream, variant):
    import flashim
    from flashim.models.lightglue import weights
    from flashim.models.weights_util import find_weights

    try:
        checkpoint = find_weights(weights.CVG_CHECKPOINTS[variant])
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    extractor_name, input_dim = VARIANTS[variant]
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        extractor = flashim.load(extractor_name, "fp32")
        with torch.inference_mode():
            a, b = (extractor.extract(str(path), resize=1024, top_k=2048) for path in IMAGES)
        ours = flashim.load("lightglue", "fp32", config={"variant": variant})
        oracle = upstream(features=None, input_dim=input_dim, depth_confidence=-1, width_confidence=-1, flash=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for i in range(oracle.conf.n_layers):  # upstream's renaming of pre-``transformers.N`` keys
            for old, new in ((f"self_attn.{i}", f"transformers.{i}.self_attn"), (f"cross_attn.{i}", f"transformers.{i}.cross_attn")):
                state = {k.replace(old, new): v for k, v in state.items()}
        missing, _ = oracle.load_state_dict(state, strict=False)
        assert not [k for k in missing if not k.startswith("confidence_thresholds")]
        oracle = oracle.to(ours.device).eval()
        with torch.inference_mode():
            result = ours.match_features(a, b)
            size = lambda f: torch.tensor([[f.image_size[1], f.image_size[0]]], device=ours.device)  # (w, h)
            pred = oracle({
                "image0": {"keypoints": a.keypoints, "descriptors": a.descriptors, "image_size": size(a)},
                "image1": {"keypoints": b.keypoints, "descriptors": b.descriptors, "image_size": size(b)},
            })
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous
    expected = set(map(tuple, pred["matches"][0].tolist()))
    got = set(zip(result.indices_A.tolist(), result.indices_B.tolist()))
    assert len(expected) > 100 and got == expected
