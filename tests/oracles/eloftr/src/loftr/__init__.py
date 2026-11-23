# Vendored from https://github.com/zju3dv/EfficientLoFTR @ ffd4a46440 (src/loftr/__init__.py) as the reference implementation of FlashIM's parity tests.
# SPDX-License-Identifier: Apache-2.0 (see LICENSE in tests/oracles/eloftr/; upstream re-licensed later commits to ZJU3DV PRL v1.0).
# Changes: none besides this header.

from .loftr import LoFTR
from .utils.full_config import full_default_cfg
from .utils.opt_config import opt_default_cfg
from .loftr import reparameter