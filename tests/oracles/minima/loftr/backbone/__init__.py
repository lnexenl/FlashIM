# Vendored from https://github.com/LSXI7/LoFTR_minima (src/loftr/backbone/__init__.py; submodule of https://github.com/LSXI7/MINIMA)
# as the reference implementation of FlashIM's MINIMA-LoFTR parity tests.
# SPDX-License-Identifier: Apache-2.0 (see LICENSE in this directory).
# Changes: none besides this header.

from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
