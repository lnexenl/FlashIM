import logging

import pytest
import torch

from flashim import log
from flashim.ops._registry import OPS, DeviceSpec, Op
from flashim.platforms import Feature, Platform, Vendor, detect, nvidia, platform_for, register_detector
from flashim.platforms.nvidia import isa_features


def test_isa_features_are_not_monotonic_in_capability():
    assert {Feature.MMA_SYNC, Feature.TMA, Feature.WGMMA} <= isa_features((9, 0))
    assert Feature.TCGEN05 in isa_features((10, 0))
    sm120 = isa_features((12, 0))
    assert Feature.TMA in sm120
    assert Feature.WGMMA not in sm120 and Feature.TCGEN05 not in sm120
    assert Feature.TMA not in isa_features((8, 9))


def test_triton_is_pointed_at_a_ptxas_that_knows_the_architecture(monkeypatch, tmp_path):
    """sm_110 (Jetson AGX Thor) is unknown to the ptxas Triton 3.6 bundles: the CUDA toolkit's is used if it works."""
    pytest.importorskip("triton")
    toolkit = tmp_path / "bin" / "ptxas"
    toolkit.parent.mkdir()
    toolkit.touch()
    monkeypatch.setattr(nvidia, "nvcc_path", lambda: str(toolkit.with_name("nvcc")))
    probed = []
    knows = {str(toolkit)}
    monkeypatch.setattr(nvidia, "_ptxas_assembles", lambda ptxas, arch: probed.append((ptxas, arch)) or ptxas in knows)

    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH", raising=False)
    assert nvidia.triton_targets((12, 0)) and nvidia.triton_targets((8, 7)) and not probed  # known: no subprocess
    assert nvidia.triton_targets((11, 0))
    assert nvidia.os.environ["TRITON_PTXAS_BLACKWELL_PATH"] == str(toolkit)
    assert [arch for _, arch in probed] == ["sm_110a", "sm_110a"] and probed[0][0].endswith("ptxas-blackwell")

    monkeypatch.setenv("TRITON_PTXAS_BLACKWELL_PATH", "/user/ptxas")  # the user's choice is not overridden
    assert not nvidia.triton_targets((11, 0)) and nvidia.os.environ["TRITON_PTXAS_BLACKWELL_PATH"] == "/user/ptxas"
    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH")
    knows.clear()
    assert not nvidia.triton_targets((11, 0)) and "TRITON_PTXAS_BLACKWELL_PATH" not in nvidia.os.environ


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
