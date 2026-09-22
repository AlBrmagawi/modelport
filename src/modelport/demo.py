from pathlib import Path

from modelport.domain import BenchmarkConfig
from modelport.reporting import export_reports
from modelport.security import save_npz, write_json


def run_demo(client, output: Path, architecture: str = "vision-mlp"):
    from modelport.tensors import synthetic

    output.mkdir(parents=True, exist_ok=True)
    capabilities = client.capabilities()
    source = client.fixture(architecture)
    # Exercise local importing from actual generated bundle files.
    imported = client.import_model(client.store.path(source.artifact_id))
    data = synthetic(imported.descriptor.inputs, batch=4, seed=4242)
    save_npz(output / "validation-inputs.npz", data)
    fp32 = client.convert(client.plan(imported.artifact_id))
    optimized = client.convert(client.plan(fp32.artifact_id, optimize=True))
    quantized = client.convert(client.plan(fp32.artifact_id, precision="dynamic-int8"))
    variants = [imported, fp32, optimized, quantized]
    validations = []
    for artifact in variants[1:]:
        policy = (
            "int8-synthetic-v1"
            if artifact.descriptor.state.precision == "dynamic-int8"
            else "fp32-default"
        )
        report = client.validate(
            imported.artifact_id,
            artifact.artifact_id,
            inputs=output / "validation-inputs.npz",
            policy=policy,
        )
        report.require_passed()
        validations.append(report.model_dump(mode="json"))
    benchmarks = [
        client.benchmark(artifact.artifact_id, BenchmarkConfig()).model_dump(mode="json")
        for artifact in variants
    ]
    with client.load(fp32.artifact_id, require_validated=True) as model:
        prediction = model.predict(data)
        prediction_shapes = {name: list(value.shape) for name, value in prediction.outputs.items()}
    for artifact in variants:
        client.export_artifact(artifact.artifact_id, output / (artifact.artifact_id[:12] + ".zip"))
        write_json(output / (artifact.artifact_id[:12] + "-manifest.json"), artifact.manifest)
    export_reports(output, benchmarks)
    write_json(output / "validations.json", validations)
    write_json(output / "capabilities.json", capabilities.model_dump(mode="json"))
    summary = {
        "schema_version": 1,
        "scope": (
            "synthetic model and independent synthetic input evidence; no real-world "
            "classification accuracy"
        ),
        "source": imported.artifact_id,
        "fp32": fp32.artifact_id,
        "optimized": optimized.artifact_id,
        "dynamic_int8": quantized.artifact_id,
        "prediction_shapes": prediction_shapes,
        "validations": [r["state"] for r in validations],
        "mean_latency_ms": {r["artifact_id"]: r["measurements"]["mean_ms"] for r in benchmarks},
        "report_directory": str(output.resolve()),
    }
    write_json(output / "summary.json", summary)
    return summary
