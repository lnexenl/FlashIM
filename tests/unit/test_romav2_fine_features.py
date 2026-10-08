"""RoMaV2's VGG19-BN fine features: BatchNorm-folded and INT8 variants against the plain module (CPU, random weights)."""

import torch
from torch import nn

from flashim.models.romav2.fine_features import VGG19BN
from flashim.quant.apply import quantize_model
from flashim.quant.calibration import Int8Calibrator
from flashim.quant.methods.conv2d import StaticInt8Conv2d
from flashim.quant.scheme import INT8_ACTIVATIONS, INT8_WEIGHTS, QuantConfig, QuantScheme


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


@torch.inference_mode()
def test_folded_and_int8_fine_features_track_the_plain_module():
    torch.manual_seed(0)
    vgg = VGG19BN().eval()
    for module in vgg.modules():  # non-trivial BatchNorm statistics: the fold has something to do
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.normal_(0, 0.1)
            module.running_var.uniform_(0.5, 1.5)
            module.weight.uniform_(0.5, 1.5)
            module.bias.normal_(0, 0.1)
    image = torch.rand(2, 3, 32, 40)
    plain = vgg(image)

    assert vgg.fold_batchnorm(channels_last=True) == 8
    convs = [f"layers.{i}" for i, layer in enumerate(vgg.layers) if isinstance(layer, nn.Conv2d)]
    with Int8Calibrator(vgg, convs, percentile=1.0) as observer:
        folded = vgg(image)
    config = QuantConfig({"conv": QuantScheme(("layers.*",), INT8_WEIGHTS, INT8_ACTIVATIONS)}, name="int8")
    report = quantize_model(vgg, config, observer.table())
    assert report.convs == convs and all(isinstance(vgg.get_submodule(name), StaticInt8Conv2d) for name in convs)
    quantized = vgg(image)

    for scale, channels in ((1, 64), (2, 128), (4, 256)):
        assert quantized[scale].shape == plain[scale].shape == (2, 32 // scale, 40 // scale, channels)
        assert quantized[scale].dtype == plain[scale].dtype == torch.bfloat16
        assert (quantized[scale].float() >= 0).all()  # the ReLU runs in the quantized conv's epilogue
        assert _relative_error(folded[scale], plain[scale]) < 0.02  # BF16 rounding of the folded weights
        assert _relative_error(quantized[scale], folded[scale]) < 0.05, scale  # measured 0.005 .. 0.017
