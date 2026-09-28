"""ALIKED + LightGlue SparsePipeline wiring, no checkpoints needed."""

import torch

from flashim.models.aliked import ALIKED, ALIKEDConfig
from flashim.models.lightglue import LightGlue
from flashim.models.lightglue.config import LightGlueConfig
from flashim.models.lightglue.pipeline import SparsePipeline

ALIKED_SMALL = ALIKEDConfig(c1=4, c2=8, c3=8, c4=8, dim=8, K=3, M=4)
LIGHTGLUE_SMALL = LightGlueConfig(dim=16, input_dim=8, num_heads=2, num_layers=2, reference_size_order=False)


def test_pipeline_accepts_aliked_resize_and_top_k():
    torch.manual_seed(0)
    extractor = ALIKED(ALIKED_SMALL).eval()
    for m in extractor.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_var.uniform_(0.5, 1.5)
    matcher = LightGlue(LIGHTGLUE_SMALL).eval()
    for p in matcher.parameters():
        torch.nn.init.normal_(p, std=0.2)
    pipeline = SparsePipeline(extractor, matcher)

    image_a = torch.rand(1, 3, 130, 170)
    image_b = torch.rand(1, 3, 100, 200)
    m = pipeline.match(image_a, image_b, resize=96, top_k=16, threshold=0.0)
    assert m.keypoints_A.shape[-1] == 2 and m.keypoints_B.shape[-1] == 2
    assert len(m.confidence) == len(m.keypoints_A) == len(m.keypoints_B)
