import pytest
import torch

from flashim import ops
from flashim.ops._registry import OPS, DeviceSpec, Op
from flashim.platforms import Feature, Platform


def _op(name):
    OPS.pop(name, None)
    return Op(name, lambda x: "reference")


def test_priority_then_narrowest_range():
    op = _op("t_order")
    op.register("wide", device=DeviceSpec("cuda", (8, 0)))(lambda x: "wide")
    op.register("narrow", device=DeviceSpec("cuda", (12, 0), (12, 9)))(lambda x: "narrow")
    op.register("low", device=DeviceSpec("cuda"), priority=-1)(lambda x: "low")
    sm120 = Platform("cuda", 0, capability=(12, 0))
    sm90 = Platform("cuda", 0, capability=(9, 0))
    assert [i.name for i in op.candidates(sm120)] == ["narrow", "wide", "low"]
    assert [i.name for i in op.candidates(sm90)] == ["wide", "low"]
    assert op.candidates(Platform("cpu")) == []


def test_required_features_filter():
    op = _op("t_features")
    op.register("mx", device=DeviceSpec("cuda"), requires=[Feature.MXFP8_SCALED_MM])(lambda x: "mx")
    assert op.candidates(Platform("cuda", 0, capability=(9, 0))) == []
    assert op.candidates(Platform("cuda", 0, capability=(12, 0), features=frozenset({Feature.MXFP8_SCALED_MM})))


def test_cpu_falls_back_to_reference_and_overrides():
    op = _op("t_cpu")
    op.register("any", device=DeviceSpec("*"), when=lambda x: x.numel() > 1)(lambda x: "any")
    assert op(torch.zeros(2)) == "any"
    assert op(torch.zeros(1)) == "reference"  # predicate rejects
    ops.configure("-t_cpu")
    assert op(torch.zeros(2)) == "reference"
    ops.configure("t_cpu:any")
    assert op(torch.zeros(1)) == "any"
    ops.configure(None)
    assert op.get("reference")(torch.zeros(2)) == "reference"
    assert op.get("any")(torch.zeros(2)) == "any"
    assert op.get("any")(torch.zeros(1)) == "reference"
    with pytest.raises(ValueError):
        op.get("missing")
    with pytest.raises(ValueError):
        ops.configure("bogus")


def test_strict_mode_raises_off_cpu():
    op = _op("t_strict")
    ops.set_strict(True)
    assert op(torch.zeros(1)) == "reference"  # CPU is allowed to use references
    if torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            op(torch.zeros(1, device="cuda"))
