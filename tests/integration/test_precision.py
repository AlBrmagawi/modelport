import numpy as np
import pytest

from modelport.errors import ModelPortError
from modelport.tensors import synthetic


def test_real_fp16_preserves_io_and_validates_against_original(workflow):
    import onnx

    client, source = workflow["client"], workflow["fp32"]
    target = client.convert(client.plan(source.artifact_id, precision="fp16"))
    graph = onnx.load(client.store.path(target.artifact_id) / "model.onnx")
    assert any(t.data_type == onnx.TensorProto.FLOAT16 for t in graph.graph.initializer)
    assert target.validation_state == "validated"
    assert target.descriptor.state.precision == "fp16"
    assert target.descriptor.inputs[0].dtype == "float32"
    report = client.repository.list_reports(target.artifact_id, "validation")[0]
    assert report["source_id"] == workflow["source"].artifact_id
    assert report["policy"]["max_normalized_l2"] == 0.005
    with client.load(target.artifact_id) as model:
        assert model.predict(synthetic(target.descriptor.inputs, 7, 44)).outputs[
            "logits"
        ].shape == (7, 10)


def test_static_int8_with_separate_datasets_and_overlap_rejection(workflow, tmp_path):
    import onnx

    client, source = workflow["client"], workflow["fp32"]
    calibration_path, validation_path = tmp_path / "calibration.npz", tmp_path / "validation.npz"
    np.savez(
        calibration_path,
        images=np.random.default_rng(505).standard_normal((512, 1, 8, 8)).astype(np.float32),
    )
    np.savez(
        validation_path,
        images=np.random.default_rng(9999).standard_normal((4, 1, 8, 8)).astype(np.float32),
    )
    calibration = client.import_dataset(calibration_path, purpose="calibration")
    validation = client.import_dataset(validation_path, purpose="validation")
    plan = client.plan(
        source.artifact_id,
        precision="static-int8",
        calibration_id=calibration.dataset_id,
        validation_id=validation.dataset_id,
    )
    target = client.convert(plan)
    graph = onnx.load(client.store.path(target.artifact_id) / "model.onnx")
    assert {"QuantizeLinear", "DequantizeLinear"} <= {n.op_type for n in graph.graph.node}
    assert any(
        t.data_type == onnx.TensorProto.INT8 and len(t.dims) == 2 for t in graph.graph.initializer
    )
    assert target.validation_state == "validated"
    assert target.manifest["calibration"]["sha256"] == calibration.sha256
    report = client.repository.list_reports(target.artifact_id, "validation")[0]
    assert report["dataset"]["sha256"] == validation.sha256
    assert report["source_id"] == workflow["source"].artifact_id
    assert report["policy"]["name"] == "int8-calibrated-v1"
    client.validate(
        workflow["source"].artifact_id,
        target.artifact_id,
        validation_id=validation.dataset_id,
        policy="int8-calibrated-v1",
    ).require_passed()
    duplicate = client.import_dataset(calibration_path, purpose="validation")
    with pytest.raises(ModelPortError, match="identical samples"):
        client.plan(
            source.artifact_id,
            precision="static-int8",
            calibration_id=calibration.dataset_id,
            validation_id=duplicate.dataset_id,
        )
    with pytest.raises(ModelPortError, match="requires registered"):
        client.plan(source.artifact_id, precision="static-int8")
    (client.settings.data_dir / "datasets" / validation.sha256 / "inputs.npz").write_bytes(
        b"tampered"
    )
    with pytest.raises(ModelPortError, match="identity"):
        client.validate(
            workflow["source"].artifact_id,
            target.artifact_id,
            validation_id=validation.dataset_id,
            policy="int8-calibrated-v1",
        )
