from pathlib import Path

import pytest

from modelport.sdk import ModelPort


@pytest.fixture(scope="session")
def workflow(tmp_path_factory):
    directory = tmp_path_factory.mktemp("real-cpu-workflow")
    with ModelPort.local(directory) as client:
        source = client.fixture()
        capabilities = client.doctor()
        assert all(
            capabilities.probes[name]["passed"]
            for name in ("torch-onnx", "onnx-optimize", "onnx-dynamic-int8")
        )
        fp32 = client.convert(client.plan(source.artifact_id))
        optimized = client.convert(client.plan(fp32.artifact_id, optimize=True))
        int8 = client.convert(client.plan(fp32.artifact_id, precision="dynamic-int8"))
        yield {
            "client": client,
            "source": source,
            "fp32": fp32,
            "optimized": optimized,
            "int8": int8,
            "directory": directory,
        }


@pytest.fixture
def client(tmp_path: Path):
    with ModelPort.local(tmp_path / "data") as service:
        yield service
