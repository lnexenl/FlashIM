"""RaCo without a checkpoint: layout, keypoint selection and the extractor wiring on CPU."""

import pytest
import torch

from flashim.models.aliked import ALIKED, ALIKEDConfig
from flashim.models.raco import RaCo, RaCoConfig
from flashim.models.raco import keypoints as kp
from flashim.models.raco.extractor import RaCoALIKED
from flashim.models.raco.fuse import fold_batchnorm

SMALL = RaCoConfig(num_keypoints=32, candidate_multiplier=2)


def _model(config=SMALL):
    torch.manual_seed(0)
    model = RaCo(config).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_var.uniform_(0.5, 1.5)
            module.running_mean.normal_(std=0.3)
    return model


def test_state_dict_matches_release_layout():
    keys = RaCo().state_dict().keys()
    assert "block2.match_dims.weight" in keys and "score_head.6.weight" in keys
    assert sum(k.startswith("ranker_head.") and k.endswith("match_dims.weight") for k in keys) == 9
    assert "ranker_head.9.weight" in keys  # the 5x5 head


def test_candidate_pool_follows_config():
    assert RaCoConfig(num_keypoints=1024).num_candidates == 2048
    assert RaCoConfig(num_keypoints=3000).num_candidates == 3840  # capped
    assert RaCoConfig(num_keypoints=256, candidate_multiplier=1).num_candidates == 256


def test_detect_candidates_finds_the_local_maxima():
    logits = torch.zeros(1, 1, 16, 16)
    logits[0, 0, 5, 7], logits[0, 0, 11, 3] = 2.0, 1.0
    found = kp.detect_candidates(logits, 2, radius=3, temperature=0.5)[0]
    assert set(map(tuple, found.round().int().tolist())) == {(7, 5), (3, 11)}  # (x, y)


def test_keypoints_are_ranked_best_first_and_in_bounds():
    model = _model()
    keypoints, scores = model.extract_keypoints(torch.rand(1, 3, 64, 96))
    assert keypoints.shape == (1, 32, 2) and scores.shape == (1, 32)
    assert (scores[0, :-1] >= scores[0, 1:]).all()
    assert (keypoints[..., 0] < 96 + 1).all() and (keypoints[..., 1] < 64 + 1).all() and (keypoints >= 0).all()


def test_fold_batchnorm_keeps_the_selected_keypoints():
    torch.manual_seed(1)
    model = _model()
    image = torch.rand(1, 3, 64, 64)
    before, _ = model.extract_keypoints(image)
    assert fold_batchnorm(model) == 1 + 3 + 9  # block1 (ConvBlock), block2-4 and the ranker (ResBlocks)
    after, _ = model.extract_keypoints(image)
    a, b = (set(map(tuple, k[0].round().int().tolist())) for k in (before, after))
    assert len(a & b) / len(a) > 0.9


def test_extractor_and_alignment_error():
    model = _model()
    aliked = ALIKED(ALIKEDConfig(c1=4, c2=8, c3=8, c4=8, dim=8, K=3, M=4)).eval()
    extractor = RaCoALIKED(model, aliked)
    features = extractor.extract(torch.rand(1, 3, 64, 96), top_k=16)
    assert features.keypoints.shape == (1, 16, 2) and features.descriptors.shape == (1, 16, 8)
    assert features.image_size == (64, 96)
    with pytest.raises(ValueError, match="32-aligned"):
        extractor.extract(torch.rand(1, 3, 50, 60))
    resized = extractor.extract(torch.rand(1, 3, 50, 60), resize=64, top_k=16)
    assert resized.image_size == (50, 60)
    matches = extractor.match(torch.rand(1, 3, 64, 96), torch.rand(1, 3, 64, 96), max_matches=16)
    assert matches.keypoints_A.shape == matches.keypoints_B.shape and len(matches) <= 16
    assert matches.features_A.keypoints.shape == (1, 16, 2)
