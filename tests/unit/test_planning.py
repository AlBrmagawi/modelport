import pytest

from modelport.domain import (
    ArtifactBundle,
    CapabilitySnapshot,
    ConversionRequest,
    ModelDescriptor,
    ModelState,
)
from modelport.errors import ModelPortError
from modelport.planning import ADAPTERS, assert_plan_current, build_plan


def inputs():
    state = ModelState(
        format="pytorch-bundle",
        precision="fp32",
        architecture="vision-mlp",
        architecture_version="1",
    )
    source = ArtifactBundle(
        artifact_id="a" * 64,
        digest="a" * 64,
        files=[],
        descriptor=ModelDescriptor(name="fixture", state=state, runtime_compatibility="compatible"),
        manifest={},
    )
    snapshot = CapabilitySnapshot(
        fingerprint="test",
        checked_at="now",
        packages={},
        environment={},
        providers=["CPUExecutionProvider"],
        probes={},
        adapters=[
            {**adapter, "availability": "available", "probe_verified": True} for adapter in ADAPTERS
        ],
    )
    return source, snapshot


def test_bounded_deterministic_path_no_cycles():
    source, snapshot = inputs()
    request = ConversionRequest(
        source_id=source.artifact_id, precision="dynamic-int8", optimize=True
    )
    plans = [build_plan(source, request, snapshot) for _ in range(5)]
    assert len({plan.plan_id for plan in plans}) == 1
    assert [step.adapter_id for step in plans[0].steps] == [
        "torch-onnx",
        "onnx-optimize",
        "onnx-dynamic-int8",
    ]
    assert len({step.adapter_id for step in plans[0].steps}) == 3


def test_unverified_planned_and_unavailable_are_ineligible():
    source, snapshot = inputs()
    snapshot.adapters[0]["probe_verified"] = False
    with pytest.raises(ModelPortError, match="No verified route"):
        build_plan(source, ConversionRequest(source_id=source.artifact_id), snapshot)


def test_stale_options_and_no_state_change():
    source, snapshot = inputs()
    plan = build_plan(source, ConversionRequest(source_id=source.artifact_id), snapshot)
    plan.steps[0].options = {"opset": 999}
    with pytest.raises(ModelPortError, match="changed"):
        assert_plan_current(plan, source, snapshot)
    source.descriptor.state = ModelState(format="onnx", precision="fp32")
    with pytest.raises(ModelPortError, match="already"):
        build_plan(source, ConversionRequest(source_id=source.artifact_id), snapshot)


def test_unbound_weights_and_unsupported_device():
    source, snapshot = inputs()
    source.descriptor.state = ModelState(format="safetensors")
    with pytest.raises(ModelPortError, match="No verified route"):
        build_plan(source, ConversionRequest(source_id=source.artifact_id), snapshot)
