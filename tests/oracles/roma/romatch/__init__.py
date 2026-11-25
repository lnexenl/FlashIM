# Vendored from https://github.com/Parskatt/RoMa (romatch/__init__.py) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2023 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.
# Note: romatch/models/transformer/ is DINOv2 code (Meta, Apache-2.0: LICENSE_DINOv2); files keep their own Meta headers.

import os
from .models import roma_outdoor, tiny_roma_v1_outdoor, roma_indoor

DEBUG_MODE = False
RANK = int(os.environ.get('RANK', default = 0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1