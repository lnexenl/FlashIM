# Vendored from https://github.com/castacks/UniCeption @ ee7fa0b (uniception/models/prediction_heads/__init__.py) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2024, AirLab Stacks. SPDX-License-Identifier: BSD-3-Clause (see LICENSE_UniCeption in this directory).
# Changes: none besides this header.

"""
Init UniCeption Prediction Heads
"""

from uniception.models.prediction_heads.base import (
    AdaptorInput,
    AdaptorOutput,
    Covariance2DAdaptorOutput,
    MaskAdaptorOutput,
    PredictionHeadInput,
    PredictionHeadOutput,
    RegressionAdaptorOutput,
    RegressionWithConfidenceAdaptorOutput,
    RegressionWithConfidenceAndMaskAdaptorOutput,
    RegressionWithMaskAdaptorOutput,
    UniCeptionAdaptorBase,
    UniCeptionPredictionHeadBase,
)
