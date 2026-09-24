"""FlashIM's native RoMaV2 against the RoMaV2 reference package (same process, same inputs).

BF16 must be bit-identical to ``romav2.RoMaV2``; FP8 presets must be bit-identical to
``romav2.fast_fp8.FastFP8Model`` with the matching refiner variant and calibration.
"""

import pytest
import torch

from ...conftest import ASSETS, ROMAV2_ROOT

pytestmark = [pytest.mark.cuda, pytest.mark.weights, pytest.mark.oracle, pytest.mark.slow]

PAIR = (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")
CAL = ROMAV2_ROOT / "data/quantization/romav2_full_fp8_mega1500.pt"
VARIANTS = {"fp8-p8": "none", "fp8-s4": "s4_fused", "fp8": "s4_fused_s2_fused"}


def _compare(ref: dict, out: dict) -> list[str]:
    keys = ["warp_AB", "confidence_AB", "warp_BA", "confidence_BA"]
    mismatched = [k for k in keys if ref[k] is not None and not torch.equal(ref[k], out[k])]
    for stride in (4, 2, 1):
        for direction in ("AB", "BA"):
            r, o = ref[f"refiner_{stride}_{direction}"], out[f"refiner_{stride}_{direction}"]
            if r is not None and not torch.equal(r["warp"], o["warp"]):
                mismatched.append(f"refiner_{stride}_{direction}")
    for key in ("warp_AB", "attn_AB_logits"):
        if not torch.equal(ref["matcher"][key], out["matcher"][key]):
            mismatched.append(f"matcher.{key}")
    return mismatched


def _oracle(setting: str):
    from romav2 import RoMaV2 as Oracle

    oracle = Oracle(Oracle.Cfg(compile=False))
    oracle.apply_setting(setting)
    oracle._batch_refiner_pair_branches = True
    return oracle


@pytest.mark.parametrize("setting", ["turbo", "mega1500"])
def test_bf16_bit_exact(setting, fp32_highest):
    from flashim.models.romav2 import RoMaV2, RoMaV2Config

    oracle = _oracle(setting)
    model = RoMaV2.from_pretrained(RoMaV2Config(setting=setting))
    inputs = model.prepare(*PAIR)
    with torch.inference_mode():
        ref = oracle(*inputs)
        out = model(*inputs)
    assert _compare(ref, out) == []


@pytest.mark.sm120
@pytest.mark.skipif(not CAL.exists(), reason="RoMaV2 calibration file not present")
@pytest.mark.parametrize("preset", ["fp8-p8", "fp8"])
def test_fp8_bit_exact(preset, fp32_highest):
    from romav2.fast_fp8 import FastFP8Model
    from romav2.full_fp8_runtime import FullFP8Calibration
    from romav2.staged_fp8 import apply_refiner_variant

    from flashim.models.romav2.build import load_romav2

    calibration = FullFP8Calibration.load(CAL)
    oracle = FastFP8Model(_oracle("mega1500"), calibration, keep_bf16_modules=("refiners",))
    apply_refiner_variant(oracle.model, calibration, VARIANTS[preset])
    model = load_romav2(preset, setting="mega1500")
    summary = model.quant_report.summary()
    # 146 RoMaV2 linears minus the 24 in DINOv3 blocks 18-23, which FlashIM truncates.
    assert summary["linears"] == 122 and summary["linear_kernels"] == {"cublas_mxfp8": 122}
    inputs = model.prepare(*PAIR)
    with torch.inference_mode():
        ref = oracle(*inputs)
        out = model(*inputs)
    assert oracle.last_dispatch_stats["fallback_calls"] == 0
    assert _compare(ref, out) == []
