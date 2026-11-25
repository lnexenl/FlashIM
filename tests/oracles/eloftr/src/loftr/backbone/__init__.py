# Vendored from https://github.com/zju3dv/EfficientLoFTR @ ffd4a46440 (src/loftr/backbone/__init__.py) as the reference implementation of FlashIM's parity tests.
# SPDX-License-Identifier: Apache-2.0 (see LICENSE in tests/oracles/eloftr/; upstream re-licensed later commits to ZJU3DV PRL v1.0).
# Changes: none besides this header.

from .backbone import RepVGG_8_1_align

def build_backbone(config):
    if config['backbone_type'] == 'RepVGG':
        if config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return RepVGG_8_1_align(config['backbone'])
        else:
            raise ValueError(f"LOFTR.ALIGN_CORNER {config['align_corner']} not supported.")
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
