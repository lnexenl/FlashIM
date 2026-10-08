# Vendored from https://github.com/fabio-sim/LightGlue-ONNX (lightglue_dynamo/ops/shape_utils.py) as the reference implementation of FlashIM's parity tests.
# Copyright the LightGlue-ONNX authors (derived from cvg/RaCo, cvg/LightGlue). SPDX-License-Identifier: Apache-2.0 (see LICENSE in this directory).
# Changes: none besides this header.

import torch


def shape_as_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return dynamic shape as a tensor when exporting; otherwise use eager shape."""
    if torch.onnx.is_in_onnx_export() and hasattr(torch, "_shape_as_tensor"):
        return torch._shape_as_tensor(tensor)
    return torch.tensor(tensor.shape, device=tensor.device)
