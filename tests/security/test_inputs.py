import io
import json
import struct
import zipfile

import numpy as np
import pytest

from modelport.domain import ArchitectureConfig, ModelBundleSpec
from modelport.errors import ModelPortError
from modelport.reporting import benchmark_html, csv_safe
from modelport.security import (
    child_environment,
    load_npz,
    read_json,
    safe_relative,
    safetensors_header,
)


@pytest.mark.parametrize(
    "path",
    [
        "../evil",
        "a/../../evil",
        "/root",
        "C:/x",
        "C:x",
        "\\\\server\\share",
        "a\\..\\b",
        "foo:stream",
        "CON",
        "nul.txt",
        "com1.bin",
        "a.",
        "a ",
        "a//b",
        "./x",
        "a/../b",
    ],
)
def test_cross_platform_unsafe_paths(path):
    with pytest.raises(ModelPortError):
        safe_relative(path)


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".bin", ".py", ".zip", ".tar", ".dll"])
def test_unsafe_serialization_rejected_without_loading(client, tmp_path, suffix):
    path = tmp_path / ("evil" + suffix)
    path.write_bytes(b"not to be deserialized")
    with pytest.raises(ModelPortError, match="pickle"):
        client.import_model(path)


def test_lfs_pointer_and_oversized_file(client, tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ModelPortError, match="LFS"):
        client.import_model(path)
    client.settings.max_file_bytes = 8
    with pytest.raises(ModelPortError, match="limits"):
        client.import_model(path)


def test_safetensors_truncated_and_allocation_budget(tmp_path):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(struct.pack("<Q", 2**50))
    with pytest.raises(ModelPortError, match="budget"):
        safetensors_header(path, 1024, 1024)
    header = json.dumps(
        {"x": {"dtype": "F32", "shape": [1000000], "data_offsets": [0, 4000000]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ModelPortError, match="truncated"):
        safetensors_header(path, 1024, 1024)


def test_npz_objects_and_claimed_allocation_rejected(tmp_path):
    path = tmp_path / "inputs.npz"
    np.savez(path, x=np.array([{"execute": "no"}], dtype=object))
    with pytest.raises(ModelPortError):
        load_npz(path)
    data = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        data, {"shape": (2**45,), "fortran_order": False, "descr": "<f4"}
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.npy", data.getvalue())
    with pytest.raises(ModelPortError):
        load_npz(path)


def test_duplicate_metadata_and_registry_configuration(tmp_path):
    from pydantic import ValidationError

    path = tmp_path / "config.json"
    path.write_text('{"architecture":"vision-mlp","architecture":"evil"}')
    with pytest.raises(ModelPortError):
        read_json(path)
    with pytest.raises(ValidationError):
        ArchitectureConfig(hidden=1_000_000)
    with pytest.raises(ValidationError):
        ModelBundleSpec(architecture="os.system", inputs=[], outputs=[])


def test_child_environment_removes_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("MODELPORT_TOKEN", "test-secret")
    env = child_environment(tmp_path)
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "MODELPORT_TOKEN" not in env
    assert "test-secret" not in env.values()


def test_report_escaping_and_csv_formula_defense():
    assert csv_safe("=HYPERLINK('bad')").startswith("'")
    report = {
        "artifact_id": "<script>alert(1)</script>",
        "report_id": "evil",
        "validation_state": "<img src=x>",
        "config": {"batch_size": 1, "threads": 1},
        "measurements": {
            "sample_count": 3,
            "mean_ms": 1,
            "p95_ms": 1,
            "throughput_items_per_second": 1,
        },
        "artifact_bytes": 1,
    }
    result = benchmark_html([report])
    assert "<script>alert" not in result
    assert "<img src=x>" not in result
    assert "&lt;script&gt;" in result


def test_npz_roundtrip_reserved_numpy_argument_names(tmp_path):
    import numpy as np

    from modelport.security import load_npz, save_npz

    arrays = {"file": np.ones((2, 3), np.float32), "allow_pickle": np.zeros((2,), np.int64)}
    path = tmp_path / "reserved.npz"
    save_npz(path, arrays)
    actual = load_npz(path)
    assert set(actual) == set(arrays)
    for name, array in arrays.items():
        np.testing.assert_array_equal(actual[name], array)
