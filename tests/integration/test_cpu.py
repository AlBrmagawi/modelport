import shutil

import numpy as np
import pytest

from modelport.domain import BenchmarkConfig
from modelport.errors import ModelPortError
from modelport.sdk import ModelPort
from modelport.tensors import synthetic

pytestmark = pytest.mark.integration


def test_real_conversion_quantization_and_original_reference(workflow):
    client, source = workflow["client"], workflow["source"]
    for name in ("fp32", "optimized", "int8"):
        artifact = workflow[name]
        assert artifact.validation_state == "validated"
        reports = client.repository.list_reports(artifact.artifact_id, "validation")
        assert reports[0]["source_id"] == source.artifact_id
        assert reports[0]["dataset"]["sample_count"] == 6
        assert all(row["passed"] for row in reports[0]["outputs"])
    evidence = workflow["int8"].manifest["steps"][-1]["evidence"]
    assert evidence["quantized_nodes"] == 2
    assert evidence["operators_after"]["MatMulInteger"] == 2
    assert (
        workflow["optimized"].manifest["steps"][-1]["evidence"]["nodes_after"]
        < workflow["optimized"].manifest["steps"][-1]["evidence"]["nodes_before"]
    )


def test_unified_loading_dynamic_batch_and_cleanup(workflow):
    import psutil

    client = workflow["client"]
    for name in ("source", "fp32", "int8"):
        artifact = workflow[name]
        with client.load(artifact.artifact_id) as model:
            pid = model.child.process.pid
            for batch in (1, 4):
                assert model.predict(synthetic(artifact.descriptor.inputs, batch, 33)).outputs[
                    "logits"
                ].shape == (batch, 10)
            with pytest.raises(ModelPortError, match="between"):
                model.predict({"images": np.zeros((65, 1, 8, 8), np.float32)})
            with pytest.raises(ModelPortError, match="casts"):
                model.predict({"images": np.zeros((4, 1, 8, 8), np.float64)})
        assert not psutil.pid_exists(pid)
        with pytest.raises(ModelPortError, match="new loaded"):
            model.predict({})


def test_dual_input_multi_output_and_batch_bounds(workflow):
    client = workflow["client"]
    source = client.fixture("dual-input")
    target = client.convert(client.plan(source.artifact_id))
    assert len(target.descriptor.inputs) == len(target.descriptor.outputs) == 2
    with client.load(target.artifact_id) as model:
        for batch in (2, 7):
            result = model.predict(synthetic(source.descriptor.inputs, batch, 321))
            assert set(result.outputs) == {"logits", "features"}
            assert result.outputs["features"].shape == (batch, 128)
        with pytest.raises(ModelPortError, match="Shared dimension"):
            model.predict(
                {"left": np.zeros((2, 64), np.float32), "right": np.zeros((3, 64), np.float32)}
            )


def test_trusted_sdk_hooks_and_failed_postprocess_do_not_reuse_previous_prediction(workflow):
    client, artifact = workflow["client"], workflow["fp32"]
    values = synthetic(artifact.descriptor.inputs, 2, 15)
    next_values = {name: value + 1 for name, value in values.items()}
    with client.load(artifact.artifact_id) as plain:
        previous = plain.predict(values).outputs["logits"]
        expected = plain.predict(next_values).outputs["logits"]
        assert not np.array_equal(previous, expected)

    calls = 0

    def postprocess(outputs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("controlled trusted callback failure")
        return {"scores": outputs["logits"] + 1}

    with client.load(
        artifact.artifact_id,
        preprocess=lambda data: {"images": data["raw"]},
        postprocess=postprocess,
    ) as model:
        with pytest.raises(ValueError, match="controlled"):
            model.predict({"raw": values["images"]})
        np.testing.assert_array_equal(
            model.predict({"raw": next_values["images"]}).outputs["scores"], expected + 1
        )
        assert not list(model.work.glob("input-*.npz"))


def test_benchmark_real_samples_and_restart(workflow):
    client, artifact = workflow["client"], workflow["fp32"]
    report = client.benchmark(
        artifact.artifact_id, BenchmarkConfig(iterations=7, repetitions=2, warmup=2)
    )
    values = report.measurements
    assert values["sample_count"] == len(values["latency_samples_ms"]) == 14
    assert values["total_timed_ms"] == pytest.approx(sum(values["latency_samples_ms"]))
    assert values["processed_items"] == 56
    assert values["throughput_items_per_second"] == pytest.approx(
        56 / (values["total_timed_ms"] / 1000)
    )
    assert values["load_ms"] > 0 and values["sampled_peak_rss_bytes"] > 0
    with ModelPort.local(workflow["directory"]) as restarted:
        assert restarted.repository.artifact(artifact.artifact_id).digest == artifact.digest
        assert restarted.repository.report(report.report_id)["measurements"] == values


def test_actual_corrupted_weights_fail_validation(workflow, tmp_path):
    import onnx
    from onnx import numpy_helper

    client, artifact, source = workflow["client"], workflow["fp32"], workflow["source"]
    path = tmp_path / "broken.onnx"
    shutil.copyfile(client.store.path(artifact.artifact_id) / "model.onnx", path)
    graph = onnx.load(path)
    bias = next(t for t in graph.graph.initializer if t.name == "last.bias")
    bias.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(bias) + 1, bias.name))
    onnx.save(graph, path)
    broken = client.import_model(path)
    report = client.validate(source.artifact_id, broken.artifact_id)
    assert report.state == "failed" and report.failures
    with pytest.raises(ModelPortError, match="validation"):
        client.load(broken.artifact_id)
    with pytest.raises(ModelPortError):
        client.validate(source.artifact_id, artifact.artifact_id, mapping={"logits": "missing"})


def test_artifact_identity_changes_and_integrity_rejected(client, tmp_path):
    source = client.fixture()
    path = client.store.path(source.artifact_id) / "weights.safetensors"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ModelPortError, match="identity"):
        client.load(source.artifact_id)
