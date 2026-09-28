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


def test_dataclass_arguments_resolve_their_device():
    from flashim.ops._registry import _first_device
    from flashim.quant.tensor import QuantizedWeight

    weight = QuantizedWeight.from_float(torch.randn(64, 64))
    assert _first_device((weight, None)) == torch.device("cpu")


@pytest.mark.cuda
def test_platform_detected_inside_compile_has_features():
    """The first ``platform_for`` call may happen while Dynamo traces; its probes must still run eagerly."""
    from flashim.platforms import detect

    detect._PLATFORMS.clear()
    compiled = torch.compile(lambda x: (x + 1, detect.platform_for(x.device))[0], backend="eager")
    compiled(torch.zeros(2, device="cuda"))
    assert detect.platform_for("cuda").supports(Feature.TRITON)


def _weighted_op(name):
    OPS.pop(name, None)
    op = Op(name, lambda x, w: ("reference", x, w), weights=1)
    prepared = []

    def prepare(w):
        prepared.append(w)
        return (w * 10,)

    op.register("packed", device=DeviceSpec("cpu"), when=lambda x, w: x > 0, prepare=prepare)(
        lambda x, packed: ("packed", x, packed)
    )
    return op, prepared


def test_bind_prepares_once_and_keeps_dispatch():
    op, prepared = _weighted_op("t_bind")
    bound = op.bind(torch.tensor(2))
    assert bound.primary == "packed"
    assert bound(torch.tensor(1))[0] == "packed" and bound(torch.tensor(3))[2] == 20
    assert len(prepared) == 1  # prepared once, reused
    # ``when`` sees the raw weight and falls back to the reference, which also gets the raw weight
    name, _, weight = bound(torch.tensor(-1))
    assert name == "reference" and weight == 2
    assert op.calls == {"packed": 2, "reference": 1}


def test_unbound_call_prepares_every_time_and_pinning_works():
    op, prepared = _weighted_op("t_unbound")
    assert op(torch.tensor(1), torch.tensor(2))[2] == 20
    assert op(torch.tensor(1), torch.tensor(2))[0] == "packed"
    assert len(prepared) == 2
    assert op.bind(torch.tensor(2), impl="reference")(torch.tensor(1))[0] == "reference"
    with pytest.raises(ValueError, match="weight operands"):
        op.bind()
    with pytest.raises(ValueError, match="no impl 'nope'"):
        op.bind(torch.tensor(2), impl="nope")


def test_prepare_needs_weight_operands():
    op = _op("t_noweights")
    with pytest.raises(ValueError, match="no weight operands"):
        op.register("x", device=DeviceSpec("cpu"), prepare=lambda: ())
