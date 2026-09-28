"""RDD without a checkpoint: module layout, key conversion and the sparse API on CPU."""

import torch

from flashim.models.rdd import RDD, RDDConfig
from flashim.models.rdd.model import target_size
from flashim.models.rdd.weights import convert_state_dict

SMALL = RDDConfig(backbone_depths=(1, 1, 1, 1), num_layers=1, top_k=64)


def test_upstream_keys_convert_to_module_paths():
    upstream = {
        "descriptor.backbone.0.encoder.model.stages.0.0.pwconv1.weight": 0,
        "descriptor.transformer.encoder.layers.0.self_attn.value_proj.weight": 1,
        "descriptor.input_proj.4.0.weight": 2,
        "descriptor.matchibility_head.0.layer.0.weight": 3,
        "detector.block1.conv1.weight": 4,
    }
    assert list(convert_state_dict(upstream)) == [
        "backbone.stages.0.0.pwconv1.weight",
        "transformer.encoder.layers.0.self_attn.value_proj.weight",
        "input_proj.4.0.weight",
        "matchibility_head.0.layer.0.weight",
        "detector.block1.conv1.weight",
    ]


def test_state_dict_matches_release_layout():
    keys = RDD().state_dict().keys()
    assert sum(1 for k in keys if k.startswith("transformer.encoder.layers.")) == 4 * 16
    assert "transformer.level_embed" in keys and "detector.convhead2.0.weight" in keys


def test_target_size():
    assert target_size(1200, 1600, 1600) == (1184, 1600)
    assert target_size(1600, 1200, 1600) == (1600, 1184)
    assert target_size(479, 641, None) == (448, 640)


def test_extract_and_match_on_cpu():
    torch.manual_seed(0)
    model = RDD(SMALL).eval()
    for module in model.modules():  # BatchNorm running stats of an untrained model
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_var.fill_(1.0)
    image = torch.rand(1, 3, 100, 130)
    with torch.inference_mode():
        a = model.extract(image)
        b = model.extract(image.flip(-1))
        m = model.match_features(a, a, threshold=0.0)
    assert a.keypoints.shape == (1, 64, 2) and a.descriptors.shape == (1, 64, 256)
    assert torch.allclose(a.descriptors.norm(dim=-1), torch.ones(1, 64), atol=1e-4)
    assert (a.keypoints[..., 0] < 130).all() and (a.keypoints[..., 1] < 100).all()
    assert b.image_size == (100, 130)
    assert len(m.confidence) == 64  # every keypoint is its own mutual nearest neighbour
