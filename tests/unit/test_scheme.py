import pytest

from flashim.models.romav2.quant import PRESETS, PROTECTED
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
