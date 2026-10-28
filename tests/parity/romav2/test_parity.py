"""FlashIM's native RoMaV2 against the RoMaV2 reference package (same process, same inputs).

BF16 must be bit-identical to ``romav2.RoMaV2``; FP8 presets must be bit-identical to
``romav2.fast_fp8.FastFP8Model`` with the matching refiner variant and calibration. FlashIM's ``local_corr``
runs the oracle's own correlation kernel here (its Triton impl is checked against the reference in tests/ops).
"""

import pytest
import torch

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.ops.correlation import local_window

from ...conftest import ASSETS, ROMAV2_CALIBRATION

pytestmark = [pytest.mark.cuda, pytest.mark.weights, pytest.mark.oracle, pytest.mark.slow]

PAIR = (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")
CAL = ROMAV2_CALIBRATION
VARIANTS = {"fp8-p8": "none", "fp8-s4": "s4_fused", "fp8-s42": "s4_fused_s2_fused"}


def _oracle_local_corr(feature0, feature1, warp, radius):
    """``romav2.local_correlation.local_corr_wrapper``: the ``fused-local-corr`` package the oracle uses when
    installed (it is not a FlashIM dependency)."""
    import local_corr

    batch, channels, h, w = feature0.shape
    window = local_window(radius, h, w, feature0.device)
    k = window.shape[1]
    coords = (warp[..., None, :] + window[:, None, None]).reshape(batch, h * w, k, 2)
    corr = local_corr.local_corr(
        feature0.reshape(batch, channels, h * w).permute(0, 2, 1).float() / channels**0.5,
        feature1.permute(0, 2, 3, 1).clone().float(),
        coords.clone(),
        mode="bilinear",
        normalized_coords=True,
    )
    return corr.permute(0, 2, 1).reshape(batch, k, h, w)


if not any(impl.name == "oracle" for impl in ops.local_corr.impls):  # opt-in: only forced below
    ops.local_corr.register("oracle", device=DeviceSpec("cuda"), priority=-1)(_oracle_local_corr)


@pytest.fixture(autouse=True)
def _same_local_corr():
    """The oracle's kernel: ``fused-local-corr`` if installed, else its ``native_torch_local_corr`` (the same
    ``grid_sample`` math as FlashIM's reference). Overrides are reset by conftest's autouse fixture."""
    from romav2 import local_correlation

    ops.configure("local_corr:oracle" if local_correlation.local_corr is not None else "local_corr:reference")


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
@pytest.mark.parametrize("preset", ["fp8-p8", "fp8-s42"])
def test_fp8_bit_exact(preset, fp32_highest):
    from romav2.fast_fp8 import FastFP8Model
    from romav2.full_fp8_runtime import FullFP8Calibration
    from romav2.staged_fp8 import apply_refiner_variant

    import flashim
    from flashim.models.romav2.spec import ORACLE_OVERLAYS

    calibration = FullFP8Calibration.load(CAL)
    oracle = FastFP8Model(_oracle("mega1500"), calibration, keep_bf16_modules=("refiners",))
    apply_refiner_variant(oracle.model, calibration, VARIANTS[preset])
    model = flashim.load("romav2", "bf16", config=ORACLE_OVERLAYS[preset], overrides={"arch.setting": "mega1500"})
    summary = model.quant_report.summary()
    # 146 RoMaV2 linears minus the 24 in DINOv3 blocks 18-23, which FlashIM truncates.
    assert summary["linears"] == 122 and summary["linear_kernels"] == {"cublas": 122}
    inputs = model.prepare(*PAIR)
    with torch.inference_mode():
        ref = oracle(*inputs)
        out = model(*inputs)
    assert oracle.last_dispatch_stats["fallback_calls"] == 0
    assert _compare(ref, out) == []
