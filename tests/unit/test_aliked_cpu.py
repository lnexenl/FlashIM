"""ALIKED without a checkpoint: module layout, keypoint decoding and the sparse API on CPU."""

import torch
import torch.nn.functional as F

from flashim.models.aliked import ALIKED, ALIKEDConfig
from flashim.models.aliked.fuse import fold_batchnorm
from flashim.models.aliked.keypoints import detect_keypoints, mutual_nn_match, simple_nms

SMALL = ALIKEDConfig(c1=4, c2=8, c3=8, c4=8, dim=8, K=3, M=4, n_limit=64)


def test_state_dict_matches_release_layout():
    keys = ALIKED().state_dict().keys()
    assert "block3.conv1.offset_conv.weight" in keys  # block3/4 are deformable
    assert "block1.conv1.weight" in keys and "block1.conv1.offset_conv.weight" not in keys  # block1/2 are plain
    assert "desc_head.agg_weights" in keys and "score_head.6.weight" in keys


def test_simple_nms_keeps_only_local_maxima():
    scores = torch.zeros(1, 1, 5, 5)
    scores[0, 0, 2, 2] = 1.0
    scores[0, 0, 2, 3] = 0.5  # inside the suppression radius of the maximum, should be zeroed
    out = simple_nms(scores, radius=2)
    assert out[0, 0, 2, 2] == 1.0 and out[0, 0, 2, 3] == 0.0


def test_detect_keypoints_topk_mode_returns_requested_count():
    torch.manual_seed(0)
    score_map = torch.rand(1, 1, 32, 32)
    keypoints, scores, dispersity = detect_keypoints(score_map, radius=2, top_k=10)
    assert keypoints[0].shape == (10, 2) and scores[0].shape == (10,) and dispersity[0].shape == (10,)
    assert (keypoints[0].abs() <= 1.0).all()  # normalized (x, y) in [-1, 1]


def test_detect_keypoints_threshold_mode_respects_n_limit():
    score_map = torch.full((1, 1, 32, 32), 0.9)  # every pixel above threshold
    keypoints, _, _ = detect_keypoints(score_map, radius=2, top_k=-1, scores_th=0.2, n_limit=5)
    assert keypoints[0].shape == (5, 2)


def test_mutual_nn_match_is_symmetric_and_thresholded():
    desc0 = F.normalize(torch.eye(4), dim=-1)
    desc1 = F.normalize(torch.eye(4)[[1, 0, 2, 3]], dim=-1)  # 0<->1 swapped, 2 and 3 unchanged
    idx0, idx1, conf = mutual_nn_match(desc0, desc1, threshold=0.5)
    assert idx0.tolist() == [0, 1, 2, 3] and idx1.tolist() == [1, 0, 2, 3]
    assert torch.allclose(conf, torch.ones(4))


def test_fold_batchnorm_is_exact_and_idempotent():
    torch.manual_seed(0)
    model = ALIKED(SMALL).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight, std=0.5)
            torch.nn.init.normal_(module.bias, std=0.5)
            module.running_var.uniform_(0.5, 2.0)
            module.running_mean.normal_(std=0.5)
    image = torch.rand(1, 3, 64, 96)
    with torch.inference_mode():
        before = model(image)

    folded = fold_batchnorm(model)
    assert folded == ["block1", "block2", "block3", "block4"]
    assert all(isinstance(m.bn1, torch.nn.Identity) for m in (model.block1, model.block2, model.block3, model.block4))

    with torch.inference_mode():
        after = model(image)
    for a, b in zip(before, after):
        assert torch.allclose(a, b, atol=1e-5)
    assert fold_batchnorm(model) == []  # already folded: no BatchNorm modules left to fold


def test_extract_and_match_on_cpu():
    torch.manual_seed(0)
    model = ALIKED(SMALL).eval()
    for module in model.modules():  # BatchNorm running stats of an untrained model
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_var.fill_(1.0)
    image = torch.rand(1, 3, 96, 128)
    with torch.inference_mode():
        a = model.extract(image, top_k=32)
        b = model.extract(image.flip(-1), top_k=32)
        m = model.match_features(a, a, threshold=0.0)
    assert a.keypoints.shape == (1, 32, 2) and a.descriptors.shape == (1, 32, SMALL.dim)
    assert torch.allclose(a.descriptors.norm(dim=-1), torch.ones(1, 32), atol=1e-4)
    assert (a.keypoints[..., 0] < 128).all() and (a.keypoints[..., 1] < 96).all()
    assert b.image_size == (96, 128)
    assert len(m.confidence) == 32  # every keypoint is its own mutual nearest neighbour
