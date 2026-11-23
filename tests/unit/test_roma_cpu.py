"""CPU unit tests for RoMa (v1): config, architecture, sampling, Matches API, and builder passes (no weights needed)."""

import pytest
import torch

import flashim
from flashim.models.roma import RoMa, RoMaConfig, quant
from flashim.quant.scheme import MXFP8, QuantConfig, QuantScheme
from flashim.types import DenseMatchResult, Matches


def test_config_normalization_and_validation():
    cfg = RoMaConfig(coarse_res=560, upsample_res=864)
    assert cfg.coarse_res == (560, 560) and cfg.upsample_res == (864, 864)
    cfg2 = RoMaConfig(coarse_res=[448, 560], upsample_res=[640, 800])
    assert cfg2.coarse_res == (448, 560) and cfg2.upsample_res == (640, 800)
    with pytest.raises(ValueError, match="divisible by 14"):
        RoMaConfig(coarse_res=512)
    with pytest.raises(ValueError, match="coarse_res"):
        RoMaConfig(coarse_res=True)


def test_resolve_roma_and_indoor_variants():
    out_cfg = flashim.resolve("roma", "fp32")
    in_cfg = flashim.resolve("roma-indoor", "fp32")
    assert out_cfg.variant is None and in_cfg.variant == "indoor"
    assert flashim.resolve("roma", "bf16").precision.compute_dtype == "bf16"
    assert flashim.resolve("roma-indoor", "bf16").variant == "indoor"
    assert flashim.resolve("roma", "fp32", overrides={"variant": "indoor"}).variant == "indoor"


def test_protected_modules_rejected():
    with torch.device("meta"):
        model = RoMa()
    names = [name for name, _ in model.named_modules()]
    for bad in (
        "encoder.cnn.layers.0",
        "encoder.dinov2.patch_embed.proj",
        "decoder.gps.16.pos_conv",
        "decoder.embedding_decoder.to_out",
        "decoder.proj.16.0",
        "decoder.conv_refiner.16.disp_emb",
        "decoder.conv_refiner.1.out_conv",
    ):
        cfg = QuantConfig({"bad": QuantScheme((bad,), MXFP8, MXFP8)}, name="bad")
        with pytest.raises(ValueError, match="protected"):
            cfg.check_protected(names, quant.PROTECTED)


def test_sample_and_pixel_coordinates():
    model = RoMa(RoMaConfig(coarse_res=(28, 28), upsample_preds=False))
    h, w = 16, 20
    warp_ab = torch.zeros(1, h, w, 2)
    warp_ba = torch.zeros(1, h, w, 2)
    cert_ab = torch.full((1, h, w, 1), 0.8)
    cert_ba = torch.full((1, h, w, 1), 0.8)
    dense = DenseMatchResult(
        warp_AB=warp_ab,
        confidence_AB=cert_ab,
        overlap_AB=cert_ab,
        precision_AB=None,
        warp_BA=warp_ba,
        confidence_BA=cert_ba,
        overlap_BA=cert_ba,
        precision_BA=None,
    )
    gen = torch.Generator().manual_seed(0)
    matches, conf = model.sample(dense, num=64, generator=gen)
    assert matches.shape == (64, 4) and conf.shape == (64,)
    assert (conf == 1.0).all()  # thresholded above 0.05
    ka, kb = model.to_pixel_coordinates(matches, 100, 200, 120, 240)
    assert ka.shape == kb.shape == (64, 2)
    assert (ka[:, 0] >= 0).all() and (ka[:, 0] <= 200).all()
    assert (kb[:, 0] >= 0).all() and (kb[:, 0] <= 240).all()


def test_match_rejects_resize_and_none_max_matches():
    with torch.device("meta"):
        model = RoMa()
    img = torch.zeros(1, 3, 32, 32)
    with pytest.raises(ValueError, match="resize"):
        model.match(img, img, resize=640)
    with pytest.raises(ValueError, match="max_matches"):
        model._match(img, img, resize=None, max_matches=None, threshold=None)


def test_matches_type_and_forward_shapes_on_synthetic_cpu():
    # Lightweight synthetic check: verify DenseMatchResult and Matches shapes on CPU without weights.
    cfg = RoMaConfig(coarse_res=(28, 28), upsample_res=(32, 32), upsample_preds=True, symmetric=True)
    model = RoMa(cfg).eval()
    # Keep BatchNorm stats finite and non-zero for random init.
    a = torch.rand(1, 3, 40, 48)
    b = torch.rand(1, 3, 44, 52)
    out = model.match(a, b, max_matches=32)
    assert isinstance(out, Matches)
    assert isinstance(out.dense, DenseMatchResult)
    assert out.dense.warp_AB.shape == (1, 32, 32, 2)
    assert out.dense.warp_BA is not None and out.dense.warp_BA.shape == (1, 32, 32, 2)
    assert out.dense.overlap_AB.shape == (1, 32, 32, 1)
    assert out.dense.precision_AB is None
    assert out.keypoints_A.shape == out.keypoints_B.shape == (32, 2)
    assert out.confidence.shape == (32,)


def test_conv_refiner_pad_channels_exact_on_cpu():
    from flashim.models.roma.decoder import REFINER_SPECS, ConvRefiner
    from flashim.quant.passes.separable_conv_block import fold_block_batchnorm

    torch.manual_seed(0)
    ref = ConvRefiner(REFINER_SPECS["1"]).eval()
    fold_block_batchnorm(ref)
    x = torch.randn(2, 9, 8, 8)
    y = torch.randn(2, 9, 8, 8)
    warp = torch.rand(2, 2, 8, 8) * 1.6 - 0.8
    with torch.inference_mode():
        f0, c0 = ref(x, y, warp)
        assert ref.pad_channels(32)
        f1, c1 = ref(x, y, warp)
    torch.testing.assert_close(f1, f0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c1, c0, atol=1e-5, rtol=1e-5)
