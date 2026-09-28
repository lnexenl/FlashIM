import logging

import pytest
import torch

from flashim import log
from flashim.ops._registry import OPS, DeviceSpec, Op
from flashim.platforms import Feature, Platform, Vendor, detect, platform_for, register_detector
from flashim.platforms.nvidia import isa_features


def test_isa_features_are_not_monotonic_in_capability():
    assert {Feature.MMA_SYNC, Feature.TMA, Feature.WGMMA} <= isa_features((9, 0))
    assert Feature.TCGEN05 in isa_features((10, 0))
    sm120 = isa_features((12, 0))
    assert Feature.TMA in sm120
    assert Feature.WGMMA not in sm120 and Feature.TCGEN05 not in sm120
    assert Feature.TMA not in isa_features((8, 9))


def test_rocm_never_matches_cuda_kernels():
    rocm = Platform("rocm", 0, capability=None, vendor=Vendor.AMD, arch="gfx942", torch_device_type="cuda")
    assert rocm.device == torch.device("cuda", 0)
    assert not DeviceSpec("cuda").matches(rocm)
    assert not DeviceSpec("cuda", (8, 0)).matches(rocm)
    assert DeviceSpec("rocm").matches(rocm)
    assert DeviceSpec("*").matches(rocm)


def test_registered_detector_is_used():
    seen = []

    def fake(index):
        seen.append(index)
        return Platform("meta", index, name="fake", vendor=Vendor.GENERIC)

    try:
        register_detector("meta", fake)
        assert platform_for("meta:3").name == "fake" and seen == [3]
    finally:
        detect._DETECTORS.pop("meta", None)
        detect._PLATFORMS.clear()


def test_explain_reports_rejections():
    OPS.pop("t_explain", None)
    op = Op("t_explain", lambda x: "reference")
    op.register("needs_mx", device=DeviceSpec("cpu"), requires=[Feature.MXFP8_SCALED_MM])(lambda x: "mx")
    op.register("small", device=DeviceSpec("cpu"), when=lambda x: x.numel() < 4)(lambda x: "small")
    report = op.explain(torch.zeros(8))
    assert report["needs_mx"] == "missing features mxfp8_scaled_mm"
    assert report["small"] == "when() rejected the arguments"
    assert report["reference"] == "selected"
    assert op.explain(torch.zeros(2))["small"] == "selected"


def test_first_use_is_logged_once(caplog):
    OPS.pop("t_log", None)
    op = Op("t_log", lambda x: "reference")
    op.register("cpu_impl", device=DeviceSpec("cpu"))(lambda x: "impl")
    with caplog.at_level(logging.DEBUG, logger="flashim.ops"):
        op(torch.zeros(1))
        op(torch.zeros(1))
    assert [r.getMessage() for r in caplog.records] == ["op t_log -> cpu_impl"]
    assert op.calls["cpu_impl"] == 2


def test_warn_once(caplog):
    logger = log.get_logger("test")
    assert logger.name == "flashim.test"
    with caplog.at_level(logging.WARNING, logger="flashim"):
        log.warn_once(logger, "t_once", "hello %s", "a")
        log.warn_once(logger, "t_once", "hello %s", "b")
    assert [r.getMessage() for r in caplog.records] == ["hello a"]


def test_set_verbosity_per_subsystem():
    before = {n: logging.getLogger(n).level for n in ("flashim", "flashim.ops")}
    try:
        log.set_verbosity("warning,ops=debug", handler=False)
        assert logging.getLogger("flashim").level == logging.WARNING
        assert logging.getLogger("flashim.ops").level == logging.DEBUG
        with pytest.raises(ValueError):
            log.set_verbosity("loud", handler=False)
    finally:
        for name, level in before.items():
            logging.getLogger(name).setLevel(level)


@pytest.mark.cuda
def test_cuda_reference_fallback_logged_with_reasons(caplog):
    OPS.pop("t_fallback", None)
    op = Op("t_fallback", lambda x: "reference")
    op.register("never", device=DeviceSpec("cuda"), when=lambda x: False)(lambda x: "never")
    with caplog.at_level(logging.INFO, logger="flashim.ops"):
        op(torch.zeros(1, device="cuda"))
    (record,) = caplog.records
    assert "t_fallback uses the reference" in record.getMessage()
    assert "never: when() rejected the arguments" in record.getMessage()


@pytest.mark.cuda
def test_dispatch_inside_compile_does_not_graph_break():
    OPS.pop("t_compile", None)
    op = Op("t_compile", lambda x: x + 1)
    op.register("impl", device=DeviceSpec("cuda"))(lambda x: x + 2)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(lambda x: op(x) * 2)(torch.zeros(4, device="cuda"))
    assert explanation.graph_break_count == 0
