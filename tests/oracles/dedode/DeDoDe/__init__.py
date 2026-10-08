# Vendored from https://github.com/Parskatt/DeDoDe (DeDoDe/__init__.py) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2023 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.
# Note: DeDoDe/transformer/ is DINOv2 code (Meta, Apache-2.0: LICENSE_DINOv2); files keep their own Meta headers.

from .model_zoo import dedode_detector_B, dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G
DEBUG_MODE = False
