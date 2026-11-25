# Trimmed from https://github.com/castacks/UniCeption @ ee7fa0b (uniception/models/encoders/__init__.py) for the UFM parity test.
# Copyright (c) 2024, AirLab Stacks. SPDX-License-Identifier: BSD-3-Clause (see LICENSE_UniCeption in this directory).
from uniception.models.encoders.base import ViTEncoderInput, ViTEncoderOutput
from uniception.models.encoders.dinov2 import DINOv2Encoder, DINOv2IntermediateFeatureReturner

ENCODER_CONFIGS = {
    "dinov2": {
        "class": DINOv2Encoder,
        "intermediate_feature_returner_class": DINOv2IntermediateFeatureReturner,
    },
}

def feature_returner_encoder_factory(encoder_str: str, **kwargs):
    return ENCODER_CONFIGS[encoder_str]["intermediate_feature_returner_class"](**kwargs)
