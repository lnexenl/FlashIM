"""TraqPoint without a checkpoint: module layout, key conversion and the sparse API on CPU."""

import torch

from flashim.models.traqpoint import TraqPoint, TraqPointConfig
from flashim.models.traqpoint.weights import convert_state_dict

SMALL = TraqPointConfig(backbone_depths=(1, 1, 1, 1), backbone_dims=(32, 64, 128, 256), num_layers=1, top_k=64)


def test_upstream_keys_convert_to_module_paths():
    upstream = {
        "descriptor.backbone.0.dino.stages.0.0.pwconv1.weight": 0,
        "descriptor.backbone.0.dino.norm.weight": 1,
        "descriptor.backbone.0.dino.norms.3.weight": 2,
        "descriptor.transformer.encoder.layers.0.self_attn.value_proj.weight": 3,
        "descriptor.input_proj.3.0.weight": 4,
        "descriptor.matchibility_head.0.layer.0.weight": 5,
        "detector.block1.conv1.weight": 6,
        "detector.policy_head.weight": 7,
    }
    assert list(convert_state_dict(upstream)) == [
        "backbone.stages.0.0.pwconv1.weight",
        "backbone.norm.weight",
        "transformer.encoder.layers.0.self_attn.value_proj.weight",
        "input_proj.3.0.weight",
        "matchibility_head.0.layer.0.weight",
        "detector.block1.conv1.weight",
        "detector.policy_head.weight",
    ]


def test_state_dict_matches_release_layout():
    keys = TraqPoint().state_dict().keys()
    assert sum(1 for k in keys if k.startswith("transformer.encoder.layers.")) == 4 * 16
    assert "transformer.level_embed" in keys and "detector.policy_head.weight" in keys
    assert "backbone.norm.weight" in keys and "backbone.norm.bias" in keys
    assert sum(1 for k in keys if k.startswith("input_proj.")) == 4 * 4


def test_extract_and_match_on_cpu():
    torch.manual_seed(0)
    model = TraqPoint(SMALL).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_var.fill_(1.0)
    image = torch.rand(1, 3, 100, 130)
    with torch.inference_mode():
        a = model.extract(image)
        b = model.extract(image.flip(-1))
        m = model.match_features(a, a, threshold=0.0)
    assert a.keypoints.shape == (1, 64, 2) and a.descriptors.shape == (1, 64, 256)
    assert torch.allclose(a.descriptors.norm(dim=-1), torch.ones(1, 64), atol=1e-4)
    assert (a.keypoints[..., 0] <= 130).all() and (a.keypoints[..., 1] <= 100).all()
    assert b.image_size == (100, 130)
    assert len(m.confidence) == 64


def test_fold_batchnorm_preserves_outputs():
    from flashim.models.traqpoint.spec import _fold_batchnorm

    torch.manual_seed(0)
    model = TraqPoint(SMALL).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(0, 0.2)
            module.running_var.uniform_(0.5, 2.0)
            if module.affine:
                module.weight.data.uniform_(0.5, 1.5)
                module.bias.data.normal_(0, 0.1)
    image = torch.rand(1, 3, 64, 64)
    feat = torch.randn(1, SMALL.d_model, 16, 16)
    with torch.inference_mode():
        det_before = model.detector(image)
        mat_before = model.matchability(feat)
    folded = _fold_batchnorm(model, None)
    assert len(folded) == 8 + 2
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    with torch.inference_mode():
        det_after = model.detector(image)
        mat_after = model.matchability(feat)
    torch.testing.assert_close(det_after, det_before, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mat_after, mat_before, atol=1e-5, rtol=1e-5)
