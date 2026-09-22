"""Bounded JSON/file IPC. No database writes or artifact publication in native children."""

import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from modelport.config import Settings
from modelport.domain import ConversionPlan, ModelDescriptor, utcnow
from modelport.errors import ModelPortError
from modelport.ports import ExecutionContext
from modelport.processes import apply_child_limits
from modelport.security import canonical, read_json, write_json
from modelport.storage import ArtifactStore


def descriptor(info: dict[str, Any]) -> ModelDescriptor:
    return ModelDescriptor.model_validate(info["descriptor"])


def operation(request: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
    work, settings = context.work_dir, context.settings
    kind, payload = request["kind"], request["payload"]
    if kind == "dataset":
        from modelport.datasets import inspect_dataset

        context.stage("checking-dataset")
        directory = work / "dataset"
        directory.mkdir(mode=0o700)
        shutil.copyfile(Path(payload["staging_path"]) / "inputs.npz", directory / "inputs.npz")
        checked_dataset = inspect_dataset(directory / "inputs.npz", payload["purpose"])
        return {"dataset": checked_dataset.model_dump(mode="json")}
    if kind == "doctor":
        from modelport.doctor import run_doctor

        return {"capabilities": run_doctor(context)}
    if kind == "benchmark":
        from modelport.benchmarking import benchmark_model
        from modelport.domain import BenchmarkConfig

        source = payload["source"]
        context.stage("benchmarking")
        benchmark_report = benchmark_model(
            Path(source["path"]),
            descriptor(source),
            source["artifact_id"],
            source["validation_state"],
            settings,
            BenchmarkConfig.model_validate(payload["config"]),
        )
        return {"benchmark": benchmark_report.model_dump(mode="json")}
    from modelport.native import NativeModel, export_onnx, inspect, optimize_onnx, quantize_onnx
    from modelport.validation import policy_named, validate_models

    if kind == "validate":
        source, target = payload["source"], payload["target"]
        context.stage("validating")
        report = validate_models(
            Path(source["path"]),
            descriptor(source),
            source["artifact_id"],
            Path(target["path"]),
            descriptor(target),
            target["artifact_id"],
            settings,
            policy_named(payload["policy"]),
            batches=payload.get("batch_sizes"),
            seed=payload.get("seed", 2027),
            inputs_path=Path(payload["inputs_path"]) if payload.get("inputs_path") else None,
            mapping=payload.get("mapping"),
        )
        return {
            "validation": report.model_dump(mode="json"),
            "failed_validation": report.state == "failed",
        }
    if kind == "predict":
        import numpy as np

        from modelport.security import load_npz

        source = payload["source"]
        model = NativeModel(Path(source["path"]), descriptor(source), settings)
        try:
            outputs = model.predict(load_npz(Path(payload["inputs_path"])))
            # Prediction outputs are a bounded job result, not an executable artifact.
            if sum(value.nbytes for value in outputs.values()) > 1024**2:
                raise ModelPortError(
                    "RESOURCE_EXHAUSTED",
                    "JSON prediction output exceeds 1 MiB; use SDK load for larger outputs",
                )
            if any(not np.isfinite(value).all() for value in outputs.values()):
                raise ModelPortError("NONFINITE_OUTPUT", "Prediction contains nonfinite output")
            return {
                "outputs": {
                    name: {
                        "values": value.tolist(),
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                    }
                    for name, value in outputs.items()
                },
                "runtime": model.info,
            }
        finally:
            model.close()
    candidate = work / "candidate"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utcnow(),
        "producer": "modelport/0.1.0",
        "warnings": [],
        "fallbacks": [],
    }
    if kind in {"import", "fixture", "remote-import", "hub-import"}:
        context.stage("inspecting")
        origin = None
        if kind == "fixture":
            from modelport.architectures import create_fixture

            create_fixture(candidate, payload.get("architecture", "vision-mlp"))
        elif kind in {"remote-import", "hub-import"}:
            from modelport.domain import HTTPSImport, HubImport
            from modelport.remote import download_bundle, hub_request

            if kind == "hub-import":
                hub = HubImport.model_validate(payload["request"])
                remote = hub_request(hub)
            else:
                remote = HTTPSImport.model_validate(payload["request"])
            origin = download_bundle(remote, candidate, context)
            if kind == "hub-import":
                origin.update(
                    {"kind": "huggingface", "repository": hub.repository, "revision": hub.revision}
                )
        else:
            shutil.copytree(payload["staging_path"], candidate)
        found = inspect(candidate, settings)
        if payload.get("name") != "bundle.json":
            found.name = payload.get("name", found.name)
        manifest.update(
            {
                "operation": kind,
                "origin": origin
                or {
                    "kind": "synthetic fixture" if kind == "fixture" else "local/upload",
                    "name": found.name,
                },
                "architecture": found.state.architecture,
                "signatures": {
                    "inputs": [s.model_dump() for s in found.inputs],
                    "outputs": [s.model_dump() for s in found.outputs],
                },
                "steps": [],
            }
        )
        digest, files = ArtifactStore(settings).inventory(candidate)
        return {
            "candidate": {
                "directory": "candidate",
                "digest": digest,
                "files": files,
                "descriptor": found.model_dump(mode="json"),
                "manifest": manifest,
            }
        }
    if kind != "convert":
        raise ModelPortError(
            "UNSUPPORTED_OPERATION", "No worker implementation for the requested operation"
        )
    from modelport.doctor import current_fingerprint

    plan = ConversionPlan.model_validate(payload["plan"])
    if current_fingerprint(plan.capability_snapshot) != plan.capability_fingerprint:
        raise ModelPortError(
            "STALE_PLAN", "Installed packages or adapter code changed after planning"
        )
    current_path = Path(payload["source"]["path"])
    current_descriptor = descriptor(payload["source"])
    steps = []
    for index, step in enumerate(plan.steps):
        context.stage(step.adapter_id, step=index + 1, steps=len(plan.steps))
        if current_fingerprint(plan.capability_snapshot) != plan.capability_fingerprint:
            raise ModelPortError("STALE_PLAN", "Adapter prerequisites changed before execution")
        saved = payload.get("checkpoints", {}).get(str(index))
        if saved:
            current_path = Path(saved["path"])
            current_descriptor = ModelDescriptor.model_validate(saved["descriptor"])
            if current_descriptor.state != step.target:
                raise ModelPortError(
                    "STALE_CHECKPOINT", "Checkpoint state disagrees with the exact plan"
                )
            steps.append({**saved["step"], "reused_checkpoint": saved["digest"]})
            context.stage("checkpoint-reused", step=index + 1, digest=saved["digest"])
            continue
        output = candidate if index == len(plan.steps) - 1 else work / f"step-{index}"
        start, started_at = time.monotonic(), utcnow()
        if step.adapter_id == "torch-onnx":
            evidence = export_onnx(current_path, output, current_descriptor, settings)
        elif step.adapter_id == "onnx-optimize":
            evidence = optimize_onnx(current_path, output)
        elif step.adapter_id == "onnx-dynamic-int8":
            evidence = quantize_onnx(current_path, output)
        elif step.adapter_id == "onnx-static-int8":
            from modelport.native import quantize_static_onnx

            evidence = quantize_static_onnx(
                current_path,
                output,
                current_descriptor,
                Path(payload["calibration_path"]),
                plan.request.calibration_batch_size,
            )
        elif step.adapter_id == "onnx-fp16":
            from modelport.native import float16_onnx

            evidence = float16_onnx(current_path, output)
        else:
            raise ModelPortError("NO_ROUTE", "Unknown adapter in plan")
        checked = inspect(output, settings)
        # Preserve registered runtime constraints; ONNX alone cannot encode batch upper bounds.
        checked.inputs, checked.outputs = current_descriptor.inputs, current_descriptor.outputs
        checked.state = step.target
        checked.name = payload["source"]["descriptor"]["name"] + " · " + plan.request.precision
        current_path, current_descriptor = output, checked
        steps.append(
            {
                **step.model_dump(mode="json"),
                "evidence": evidence,
                "started_at": started_at,
                "duration_seconds": time.monotonic() - start,
            }
        )
        if index < len(plan.steps) - 1:
            checkpoint_digest, checkpoint_files = ArtifactStore(settings).inventory(output)
            write_json(
                work / f"checkpoint-{index}.json",
                {
                    "index": index,
                    "directory": output.name,
                    "plan_id": plan.plan_id,
                    "digest": checkpoint_digest,
                    "files": checkpoint_files,
                    "descriptor": checked.model_dump(mode="json"),
                    "step": steps[-1],
                },
            )
    context.stage("checking-output")
    # A structural checker or process exit is insufficient; run the selected CPU adapter.
    from modelport.tensors import synthetic

    runtime = NativeModel(candidate, current_descriptor, settings)
    try:
        for batch in plan.request.batch_sizes:
            runtime.predict(synthetic(current_descriptor.inputs, batch, plan.request.seed))
    finally:
        runtime.close()
    store = ArtifactStore(settings)
    _, output_files = store.inventory(candidate)
    manifest.update(
        {
            "operation": "convert",
            "source_id": payload["source"]["artifact_id"],
            "source_files": payload["source"]["files"],
            "reference_id": payload["reference"]["artifact_id"],
            "origin": payload["source"]["manifest"].get("origin"),
            "architecture": current_descriptor.state.architecture,
            "architecture_identity": plan.capability_fingerprint,
            "plan_id": plan.plan_id,
            "steps": steps,
            "requested_options": plan.request.model_dump(mode="json"),
            "effective_state": current_descriptor.state.model_dump(mode="json"),
            "packages": plan.capability_snapshot.packages,
            "environment": plan.capability_snapshot.environment,
            "output_files": output_files,
            "signatures": {
                "inputs": [s.model_dump() for s in current_descriptor.inputs],
                "outputs": [s.model_dump() for s in current_descriptor.outputs],
            },
            "preprocessing": "identity",
            "calibration": plan.datasets["calibration"].model_dump(mode="json")
            if "calibration" in plan.datasets
            else None,
            "validation_policy": plan.request.policy
            or (
                {
                    "dynamic-int8": "int8-synthetic-v1",
                    "static-int8": "int8-calibrated-v1",
                    "fp16": "fp16-default",
                }.get(plan.request.precision, "fp32-default")
            ),
            "evidence_linkage": (
                "Later immutable reports reference the complete bundle digest; reports "
                "are excluded from this manifest"
            ),
        }
    )
    write_json(candidate / "manifest.json", manifest)
    digest, files = store.inventory(candidate)
    result: dict[str, Any] = {
        "candidate": {
            "directory": "candidate",
            "digest": digest,
            "files": files,
            "descriptor": current_descriptor.model_dump(mode="json"),
            "manifest": manifest,
        }
    }
    if not plan.request.skip_validation:
        context.stage("validating-original-source")
        reference = payload["reference"]
        report = validate_models(
            Path(reference["path"]),
            descriptor(reference),
            reference["artifact_id"],
            candidate,
            current_descriptor,
            digest,
            settings,
            policy_named(manifest["validation_policy"]),
            batches=plan.request.batch_sizes,
            seed=plan.request.seed,
            inputs_path=Path(payload["validation_path"])
            if payload.get("validation_path")
            else None,
        )
        result["validation"] = report.model_dump(mode="json")
        result["failed_validation"] = report.state == "failed"
    return result


def main() -> None:
    work = Path.cwd()
    request = read_json(work / "request.json", 4 * 1024**2)
    settings = Settings.model_validate(request["settings"])
    apply_child_limits(settings.memory_limit_bytes, request["timeout_seconds"])
    deadline = time.monotonic() + request["timeout_seconds"]

    def check():
        if time.monotonic() >= deadline:
            raise ModelPortError("TIMEOUT", "Worker operation exceeded its deadline")

    def progress(stage: str, payload: dict[str, Any]):
        with (work / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical({"stage": stage, "payload": payload}) + "\n")

    try:
        result = operation(request, ExecutionContext(work, settings, progress, check))
        write_json(work / "result.json", {"ok": True, "result": result})
    except ModelPortError as exc:
        write_json(work / "result.json", {"ok": False, "error": exc.problem()})
    except ImportError as exc:
        print(type(exc).__name__, file=sys.stderr)
        write_json(
            work / "result.json",
            {
                "ok": False,
                "error": ModelPortError(
                    "MISSING_DEPENDENCY", "Install the locked CPU extra: uv sync --extra cpu"
                ).problem(),
            },
        )
    except Exception as exc:
        # Native diagnostics stay in bounded local logs.
        # Public errors never echo paths or model metadata.
        traceback.print_exc()
        code = "RESOURCE_EXHAUSTED" if isinstance(exc, MemoryError) else "NATIVE_OPERATION_FAILED"
        write_json(
            work / "result.json",
            {
                "ok": False,
                "error": ModelPortError(
                    code,
                    f"{type(exc).__name__} during native operation. "
                    "Check the bounded job log, signature and package compatibility",
                ).problem(),
            },
        )
    finally:
        sys.stdout.flush()


if __name__ == "__main__":
    main()
