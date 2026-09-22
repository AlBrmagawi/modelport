import pytest

from modelport.benchmarking import benchmark_model
from modelport.config import Settings
from modelport.domain import BenchmarkConfig, ModelDescriptor, ModelState, TensorSpec
from modelport.errors import ModelPortError


def test_fixed_batch_cannot_overstate_benchmark_throughput(tmp_path):
    descriptor = ModelDescriptor(
        name="fixed-batch",
        state=ModelState(format="onnx"),
        inputs=[TensorSpec(name="input", dtype="float32", dimensions=[1, 8])],
        outputs=[],
        runtime_compatibility="unknown_until_probe",
    )
    with pytest.raises(ModelPortError, match="fixed batch size"):
        benchmark_model(
            tmp_path,
            descriptor,
            "unused",
            "unverified",
            Settings(data_dir=tmp_path),
            BenchmarkConfig(batch_size=4),
        )
