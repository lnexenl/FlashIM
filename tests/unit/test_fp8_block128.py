"""``ops.mxfp8_linear``'s ``cublas-block128`` impl (cuBLAS blockwise FP8 on Hopper) and the layer option that enables
it (``activation_block == 128``): selection logic, weight re-encoding and the merged input exponent, all on CPU."""

import torch

from flashim import ops
from flashim.backends.torch_native import gemm as G
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature, Platform
from flashim.quant.formats import mx
from flashim.quant.methods.linear import StaticMXFP8Linear
from flashim.quant.tensor import QuantizedWeight, quantize_static

NAME = "cublas-block128"


def _impl():
    return next(i for i in ops.mxfp8_linear.impls if i.name == NAME)


def _cuda(capability, *features):
    return Platform("cuda", 0, capability=capability, features=frozenset(features))


def test_ranked_above_triton_on_hopper_only():
    hopper = _cuda((9, 0), Feature.TRITON, Feature.FP8_E4M3)
    assert [i.name for i in ops.mxfp8_linear.candidates(hopper)] == [NAME, "triton"]
    assert _impl().device == DeviceSpec("cuda", min_capability=(9, 0), max_capability=(9, 0))
    for capability in [(8, 0), (8, 9), (9, 1), (10, 0), (12, 0)]:
        platform = _cuda(capability, Feature.TRITON, Feature.FP8_E4M3, Feature.MXFP8_SCALED_MM)
        assert NAME not in [i.name for i in ops.mxfp8_linear.candidates(platform)], capability
    assert NAME not in [i.name for i in ops.mxfp8_linear.candidates(_cuda((9, 0), Feature.TRITON))]  # no FP8_E4M3
    assert NAME not in [i.name for i in ops.mxfp8_linear.candidates(Platform("cpu"))]


def _quantized(rows, channels, out_features=128):
    x = torch.randn(rows, channels)
    activation = quantize_static(x, mx.calibrate_block32_exponent(x))
    return activation, QuantizedWeight.from_float(torch.randn(out_features, channels))


def _flagged(weight, block=128):
    return QuantizedWeight(weight.values, weight.scale_exponent, weight.logical_shape, block)


def test_accepts_only_flagged_weights_that_tile():
    impl = _impl()
    ok = QuantizedWeight.from_float(torch.randn(256, 384))
    assert not impl.accepts(ok)  # default weights (activation_block 32): other layers are untouched
    assert impl.accepts(_flagged(ok))
    assert not impl.accepts(_flagged(QuantizedWeight.from_float(torch.randn(256, 160))))  # K % 128
    assert not impl.accepts(_flagged(QuantizedWeight.from_float(torch.randn(96, 256))))  # N % 128


def test_rows_and_bias_gates():
    impl = _impl()
    for rows, bias, ok in [(8, None, True), (6, None, False), (8, torch.zeros(128), False)]:
        activation, weight = _quantized(rows, 256)
        assert impl.when(activation, bias, weight) is ok, (rows, bias is not None)


def test_flagged_weight_on_cpu_uses_the_reference():
    activation, weight = _quantized(8, 256)
    weight = _flagged(weight)
    reference = ops.mxfp8_linear.reference(activation, None, weight)
    assert ops.mxfp8_linear.select(activation, None, weight)[0] == "reference"
    assert "does not match cpu" in ops.mxfp8_linear.explain(activation, None, weight)[NAME]
    assert torch.equal(ops.mxfp8_linear.get(NAME)(activation, None, weight), reference)


def test_weight_reencode_is_exact_up_to_subnormals():
    torch.manual_seed(0)
    weight = QuantizedWeight.from_float(torch.randn(256, 512) * torch.rand(256, 1) * 8)  # rows of different scales
    values, scales, _ = G._prepare_block128(weight)
    assert values.dtype == torch.float8_e4m3fn and scales.shape == (2, 4) and scales.dtype == torch.float32
    block = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    dequantized = values.float() * block
    reference = weight.dequantize()
    # the payload only shrinks; a value changes only where it falls into E4M3's subnormal range (step 2^-9 block scales)
    assert ((dequantized - reference).abs() <= 2.0**-10 * block).all()
    assert (dequantized != reference).float().mean() < 1e-2
    assert values.float().abs().max() <= 448


def test_weight_reencode_of_uniform_blocks_is_identity():
    weight = QuantizedWeight.from_float(torch.randn(128, 256))
    weight = QuantizedWeight(weight.values, weight.scale_exponent.amax().expand_as(weight.scale_exponent).clone(),
                             weight.logical_shape)
    values, scales, _ = G._prepare_block128(weight)
    assert torch.equal(values.view(torch.uint8), weight.values.view(torch.uint8))
    assert torch.equal(scales, torch.full((1, 2), 2.0 ** int(weight.scale_exponent[0, 0])))


def test_zero_blocks_stay_zero():
    weight = QuantizedWeight.from_float(torch.zeros(128, 128))  # exponents are the all-zero marker -127
    values, scales, _ = G._prepare_block128(weight)
    assert not values.float().any() and torch.equal(scales, torch.ones(1, 1))


def test_group_scale_and_per_rows_cache():
    exponent = torch.tensor([4, 4, 4, 4, -127, -127, -127, -127], dtype=torch.int8)  # second group: all zero
    assert torch.equal(G._group_scale(exponent), torch.tensor([16.0, 1.0]))
    state = G._BlockActivation()
    scale = state.get(exponent, 6)
    assert scale.shape == (2, 6) and scale.is_contiguous() and torch.equal(scale[:, 0], torch.tensor([16.0, 1.0]))
    assert state.get(exponent, 6) is scale  # cached per row count
    assert state.get(exponent.clone(), 6) is not scale  # another exponent object: converted, not cached


def _layer(in_features=256, out_features=128):
    linear = torch.nn.Linear(in_features, out_features)
    exponent = torch.tensor([-3, -2, -5, 0, 4, 4, 3, -127, -127, -127, -127, -127][: in_features // 32], dtype=torch.int8)
    return StaticMXFP8Linear.from_linear(linear, exponent)


def test_set_activation_block_merges_exponents_in_place():
    layer = _layer(256)
    buffer = layer.input_exponent
    assert layer.packed_weight().activation_block == 32
    layer.set_activation_block(128)
    assert layer.input_exponent is buffer  # in place: fused producers hold the buffer
    assert buffer.tolist() == [0, 0, 0, 0, 4, 4, 4, 4]  # max of each group of 4 (-127 only if the whole group is)
    assert layer.activation_block == 128 and layer.packed_weight().activation_block == 128


def test_all_zero_group_keeps_the_marker():
    layer = _layer(384)
    layer.set_activation_block(128)
    assert layer.input_exponent.tolist()[8:] == [-127] * 4


def test_set_activation_block_rejects_untileable_blocks():
    layer = _layer(96)  # three 32-channel blocks
    for block in (128, 48):
        try:
            layer.set_activation_block(block)
        except ValueError:
            continue
        raise AssertionError(block)


def test_merged_producer_payload_equals_block32_quantization_at_the_merged_scale():
    layer = _layer(256)
    layer.set_activation_block(128)
    x = torch.randn(16, 256) * 4
    activation = ops.quantize_mxfp8(x, layer.input_exponent)
    assert torch.equal(activation.values.view(torch.uint8), quantize_static(x, layer.input_exponent).values.view(torch.uint8))
    assert torch.equal(activation.scale_exponent, layer.input_exponent)
