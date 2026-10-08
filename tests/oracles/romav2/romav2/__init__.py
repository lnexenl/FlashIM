# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/__init__.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

import logging as _logging

from .logging import configure_logger as configure_logger
from .logging import logger as _logger

if not any(not isinstance(h, _logging.NullHandler) for h in _logger.handlers):
    configure_logger()

from .fast_fp8 import FastFP8Attention as FastFP8Attention
from .fast_fp8 import FastFP8Linear as FastFP8Linear
from .fast_fp8 import FastFP8Model as FastFP8Model
from .fast_fp8 import FastFP8PackingReport as FastFP8PackingReport
from .fast_fp8_graph import FastFP8CudaGraph as FastFP8CudaGraph
from .fast_fp8_refiner import FastFP8PointwiseConv as FastFP8PointwiseConv
from .fast_fp8_refiner import FastFP8RefinerBlock as FastFP8RefinerBlock
from .fast_fp8_refiner import select_dpt_convolution_path as select_dpt_convolution_path
from .full_fp8 import FakeQuantizedLinear as FakeQuantizedLinear
from .full_fp8 import QuantizedLayout as QuantizedLayout
from .full_fp8 import QuantizedTensor as QuantizedTensor
from .full_fp8 import QuantizedWeight as QuantizedWeight
from .full_fp8 import StaticFP8Linear as StaticFP8Linear
from .full_fp8 import calibrate_block32_scale as calibrate_block32_scale
from .full_fp8 import fake_quantize_block32 as fake_quantize_block32
from .full_fp8 import fp8_gemm as fp8_gemm
from .full_fp8 import fp8_gemm_fp32 as fp8_gemm_fp32
from .full_fp8 import quantize_block32 as quantize_block32
from .full_fp8_runtime import FullFP8Calibration as FullFP8Calibration
from .full_fp8_runtime import FullFP8Calibrator as FullFP8Calibrator
from .full_fp8_runtime import FullFP8Model as FullFP8Model
from .full_fp8_runtime import FullFP8PackingReport as FullFP8PackingReport
from .full_fp8_runtime import calibrate_full_fp8 as calibrate_full_fp8
from .graph_compile import GraphCompileReport as GraphCompileReport
from .graph_compile import GraphCompileRequest as GraphCompileRequest
from .graph_compile import (
    compile_graph_regions_for_inference as compile_graph_regions_for_inference,
)
from .graph_fusion import optimize_rope_for_inference as optimize_rope_for_inference
from .inference import InferenceBackend as InferenceBackend
from .inference import InferenceReport as InferenceReport
from .inference import PortableLinear as PortableLinear
from .inference import PortableLinearScope as PortableLinearScope
from .inference import PortablePointwiseConv2d as PortablePointwiseConv2d
from .inference import convert_for_inference as convert_for_inference
from .inference import export_portable_checkpoint as export_portable_checkpoint
from .inference import load_portable_checkpoint as load_portable_checkpoint
from .inference import (
    optimize_fine_features_for_inference as optimize_fine_features_for_inference,
)
from .inference import optimize_refiner_for_inference as optimize_refiner_for_inference
from .inference import (
    quantize_refiner_pointwise_for_inference as quantize_refiner_pointwise_for_inference,
)
from .inference import resolve_inference_backend as resolve_inference_backend
from .refiner_fused_mxfp8 import FusedRefinerMXFP8Block as FusedRefinerMXFP8Block
from .refiner_fused_mxfp8 import (
    fuse_refiner_mxfp8_for_inference as fuse_refiner_mxfp8_for_inference,
)
from .romav2 import RoMaV2 as RoMaV2
