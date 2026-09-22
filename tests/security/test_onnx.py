import numpy as np
import pytest

from modelport.errors import ModelPortError


def make_external(directory):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    tensor = numpy_helper.from_array(np.eye(4, dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])],
        "external",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    directory.mkdir()
    onnx.save_model(
        model,
        str(directory / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )


def test_external_data_inspection_and_missing_dependency(client, tmp_path):
    directory = tmp_path / "external"
    make_external(directory)
    artifact = client.import_model(directory)
    assert len(artifact.files) == 2
    assert artifact.descriptor.metadata["initializer_element_count"] == 16
    assert artifact.descriptor.metadata["external_data"][0]["location"] == "weights.data"
    (directory / "weights.data").unlink()
    with pytest.raises(ModelPortError, match="missing"):
        client.import_model(directory)


@pytest.mark.parametrize(
    "location", ["../escape.data", "C:/secret.data", "\\\\host\\secret", "weights.data:stream"]
)
def test_external_paths_never_resolved_outside_bundle(client, tmp_path, location):
    import onnx

    directory = tmp_path / "external"
    make_external(directory)
    model = onnx.load(str(directory / "model.onnx"), load_external_data=False)
    model.graph.initializer[0].external_data[0].value = location
    onnx.save(model, directory / "model.onnx")
    with pytest.raises(ModelPortError):
        client.import_model(directory)


def test_external_truncated_range(client, tmp_path):
    directory = tmp_path / "external"
    make_external(directory)
    (directory / "weights.data").write_bytes(b"tiny")
    with pytest.raises(ModelPortError, match="truncated"):
        client.import_model(directory)


def test_float64_graph_does_not_claim_fp32_or_offer_fp32_transform(client, tmp_path):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "double",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [2, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    path = tmp_path / "double.onnx"
    onnx.save(model, path)
    artifact = client.import_model(path)
    assert artifact.descriptor.state.precision == "unknown"
    assert artifact.descriptor.metadata["observed_floating_dtypes"] == ["float64"]
    with client.load(artifact.artifact_id) as loaded:
        values = np.arange(8, dtype=np.float64).reshape(2, 4)
        np.testing.assert_array_equal(loaded.predict({"x": values}).outputs["y"], values)
