"""DeDoDe's ConvRefiner fusions (``dedode.spec``: ``fold_refiner_residual``, ``int8_refiner_input``)."""

import pytest
import torch

from flashim import ops
from flashim.models.dedode import blocks


def _refiner(in_dim=48, hidden=32, out_dim=9, hidden_blocks=2):
    torch.manual_seed(0)
    refiner = blocks.ConvRefiner(in_dim, hidden, out_dim, hidden_blocks=hidden_blocks).eval()
    for module in refiner.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(0, 0.1), module.running_var.uniform_(0.5, 2)
    return refiner


def test_refiner_takes_features_and_context():
    refiner = _refiner()
    features, context = torch.randn(1, 16, 12, 14), torch.randn(1, 32, 6, 7)
    with torch.no_grad():
        out = refiner(features, context)
        want = refiner(blocks.refiner_input(features, context))
    assert torch.equal(out, want) and out.shape == (1, 9, 12, 14)
    assert blocks.Decoder.refiner_input is blocks.refiner_input


def test_fold_residual_scale_keeps_the_function():
    refiner = _refiner()
    features, context = torch.randn(1, 16, 12, 14), torch.randn(1, 32, 6, 7)
    with torch.no_grad():
        want = refiner(features, context)
        assert refiner.fold_residual_scale() and not refiner.fold_residual_scale()  # once
        out = refiner(features, context)
    torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
def test_int8_refiner_input_is_bit_identical():
    from flashim.quant.methods.int8_linear import StaticInt8Conv1x1

    refiner = _refiner(in_dim=96, hidden=64).cuda()
    features = (torch.randn(1, 64, 40, 52, device="cuda") * 2).bfloat16().contiguous(memory_format=torch.channels_last)
    context = (torch.randn(1, 32, 20, 26, device="cuda") * 2).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        refiner.block1[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(refiner.block1[0], refiner.block1[1])
        refiner.block1[1] = torch.nn.Identity()
        refiner.block1[0] = StaticInt8Conv1x1.from_conv(refiner.block1[0], torch.tensor(0.03))
        refiner.hidden_blocks.bfloat16(), refiner.block1[3].bfloat16(), refiner.out_conv.bfloat16()
        want = refiner.hidden(features, context)
        refiner.quantized_input = True
        ops.reset_call_counts()
        out = refiner.hidden(features, context)
    assert ops.call_counts()["upsample_concat_int8"] == {"triton": 1}
    assert torch.equal(out, want)
