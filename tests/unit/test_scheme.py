import pytest

from flashim.models.romav2.quant import ORACLE_PRESETS, PROTECTED
from flashim.models.romav2.quant import PRESETS as _PUBLIC
from flashim.quant.scheme import MXFP8, QuantConfig, QuantScheme


def test_scheme_matching_and_ignore():
    cfg = QuantConfig(
        {"t": QuantScheme(targets=("f.blocks.*",), weights=MXFP8)}, ignore=("f.blocks.0.*",), name="x"
    )
    assert cfg.scheme_for("f.blocks.3.attn.qkv")[0] == "t"
    assert cfg.scheme_for("f.blocks.0.attn.qkv") is None
    assert cfg.scheme_for("matcher.head.norm") is None


def test_protected_modules_are_a_hard_error():
    cfg = QuantConfig({"bad": QuantScheme(targets=("refiners.*",))}, name="bad")
    with pytest.raises(ValueError, match="protected"):
        cfg.check_protected(["refiners.4.warp_head", "refiners.4.block1"], PROTECTED)

PRESETS = {**_PUBLIC, **ORACLE_PRESETS}


@pytest.mark.parametrize("name", [p for p in PRESETS if PRESETS[p].config is not None])
def test_presets_never_target_protected(name):
    names = [
        "f.patch_embed.proj",
        "refiners.4.proj",
        "refiners.2.warp_head",
        "refiners.1.confidence_head",
        "refiners.4.disp_emb",
        "matcher.head.scratch.output_conv2.2",
    ]
    PRESETS[name].config.check_protected(names, PROTECTED)
    assert all(PRESETS[name].config.scheme_for(n) is None for n in names)


def test_stem_ignore_rules_only_affect_target_presets():
    from flashim.models.dedode.spec import SPEC as DEDODE
    from flashim.models.dedode.spec import SPEC_G as DEDODE_G
    from flashim.models.ripe.spec import SPEC as RIPE

    # RIPE fp8 skips early VGG stem convs (E4M3 3-bit mantissa floor); int8 keeps them quantized and runs only
    # bit-exact INT8 passes (omitting the 3 non-exact DeDoDe passes that shift RIPE's 0.5-threshold heatmap).
    for layer in ("net.encoder.layers.3", "net.encoder.layers.7", "net.encoder.layers.10"):
        assert RIPE.quant_configs["fp8"].scheme_for(layer) is None
        assert RIPE.quant_configs["int8"].scheme_for(layer) is not None
    assert RIPE.quant_configs["fp8"].scheme_for("net.encoder.layers.14") is not None
    assert "int8_refiner_input" in RIPE.passes and "fused_pool_epilogue" in RIPE.passes
    assert not ({"fold_refiner_residual", "int8_decoder_producers", "tensor_core_first_conv"} & set(RIPE.passes))

    # DeDoDe-B int8 skips early descriptor VGG stem convs, and omits fp8 so build.NEAREST resolves fp8 -> int8;
    # DeDoDe-G keeps both int8 and fp8 with its VGG stem quantized.
    import flashim

    for layer in ("descriptor.encoder.layers.3", "descriptor.encoder.layers.7", "descriptor.encoder.layers.10"):
        assert DEDODE.quant_configs["int8"].scheme_for(layer) is None
    assert DEDODE.quant_configs["int8"].scheme_for("detector.encoder.layers.3") is not None
    assert DEDODE.quant_configs["int8"].scheme_for("descriptor.encoder.layers.14") is not None
    assert "fp8" not in DEDODE.presets
    fallbacks = {}
    resolved = flashim.resolve("dedode", "fp8", fallbacks=fallbacks)
    assert resolved.preset == "int8" and "fp8 -> int8" in next(iter(fallbacks.values()))
    assert "fp8" in DEDODE_G.presets
    assert DEDODE_G.quant_configs["int8"].scheme_for("descriptor.encoder.vgg.layers.3") is not None
    assert DEDODE_G.quant_configs["fp8"].scheme_for("descriptor.encoder.vgg.layers.3") is not None
