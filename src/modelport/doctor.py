import importlib.metadata
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from modelport.domain import CapabilitySnapshot, utcnow
from modelport.planning import ADAPTERS, PLANNED
from modelport.security import digest_file, identity, save_npz

PACKAGES = ["torch", "onnx", "onnxruntime", "onnxscript", "safetensors", "numpy"]


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def code_identity() -> str:
    directory = Path(__file__).parent
    return identity(
        {
            name: digest_file(directory / name)
            for name in (
                "architectures.py",
                "native.py",
                "planning.py",
                "validation.py",
                "doctor.py",
                "datasets.py",
                "remote.py",
            )
        }
    )


def portable_numbers(value: Any) -> Any:
    """JSON clients may serialize integral floats as integers (including -0.0)."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: portable_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_numbers(item) for item in value]
    return value


def current_fingerprint(snapshot: CapabilitySnapshot) -> str:
    return identity(
        portable_numbers(
            {
                "packages": package_versions(),
                "code": code_identity(),
                "environment": snapshot.environment,
                "providers": snapshot.providers,
                "probes": snapshot.probes,
            }
        )
    )


def dependency_issues(versions: dict[str, str | None]) -> list[dict[str, Any]]:
    requirements = {
        (name, constraint)
        for adapter in ADAPTERS
        for name, constraint in adapter["packages"].items()
    }
    issues = []
    for name, constraint in sorted(requirements):
        installed = versions.get(name)
        try:
            compatible = installed is not None and Version(installed) in SpecifierSet(constraint)
        except InvalidVersion:
            compatible = False
        if not compatible:
            issues.append({"package": name, "installed": installed, "required": constraint})
    return issues


def run_doctor(context) -> dict[str, Any]:
    from modelport.benchmarking import environment

    versions = package_versions()
    issues = dependency_issues(versions)
    probes: dict[str, Any] = {}
    providers = []
    if all(versions.values()) and not issues:
        import onnxruntime as ort

        from modelport.architectures import create_fixture
        from modelport.native import (
            export_onnx,
            float16_onnx,
            inspect,
            optimize_onnx,
            quantize_onnx,
            quantize_static_onnx,
        )
        from modelport.validation import policy_named, validate_models

        providers = ort.get_available_providers()
        work, settings = context.work_dir, context.settings
        create_fixture(work / "fixture")
        descriptor = inspect(work / "fixture", settings)
        import numpy as np

        from modelport.tensors import synthetic

        samples = [synthetic(descriptor.inputs, 64, 505 + i) for i in range(8)]
        calibration = work / "calibration.npz"
        calibration_arrays: dict[str, Any] = {
            name: np.concatenate([sample[name] for sample in samples]) for name in samples[0]
        }
        save_npz(calibration, calibration_arrays)
        for name, action, source, output in [
            (
                "torch-onnx",
                lambda s, t: export_onnx(s, t, descriptor, settings),
                work / "fixture",
                work / "fp32",
            ),
            ("onnx-optimize", optimize_onnx, work / "fp32", work / "optimized"),
            ("onnx-dynamic-int8", quantize_onnx, work / "fp32", work / "int8"),
            (
                "onnx-static-int8",
                lambda s, t: quantize_static_onnx(s, t, descriptor, calibration, 4),
                work / "fp32",
                work / "static-int8",
            ),
            ("onnx-fp16", float16_onnx, work / "fp32", work / "fp16"),
        ]:
            context.stage("probe", adapter=name)
            try:
                evidence = action(source, output)
                target = inspect(output, settings)
                report = validate_models(
                    work / "fixture",
                    descriptor,
                    "probe-source",
                    output,
                    target,
                    "probe-target",
                    settings,
                    policy_named(
                        {
                            "onnx-dynamic-int8": "int8-synthetic-v1",
                            "onnx-static-int8": "int8-calibrated-v1",
                            "onnx-fp16": "fp16-default",
                        }.get(name, "fp32-default")
                    ),
                )
                report.require_passed()
                probes[name] = {
                    "passed": True,
                    "evidence": evidence,
                    "validation": report.state,
                    "max_normalized_l2": max(row.get("normalized_l2", 0) for row in report.outputs),
                }
            except Exception as exc:
                import traceback

                traceback.print_exc()
                probes[name] = {
                    "passed": False,
                    "error": getattr(exc, "code", type(exc).__name__),
                    "reason": (
                        "Bounded probe failed; inspect job logs and installed dependency "
                        "compatibility"
                    ),
                }
    adapters = [
        {
            **adapter,
            "availability": "available"
            if probes.get(adapter["id"], {}).get("passed")
            else "missing_dependency"
            if not all(versions.values())
            else "incompatible_dependency"
            if issues
            else "probe_failed",
            "probe_verified": probes.get(adapter["id"], {}).get("passed", False),
            "compatibility": "compatible"
            if probes.get(adapter["id"], {}).get("passed")
            else "unknown_until_probe",
            "dependency_issues": issues,
            "option_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {k: {"const": v} for k, v in adapter["options"].items()},
            },
        }
        for adapter in ADAPTERS
    ]
    adapters.extend(
        {
            "id": name,
            "implementation": "planned",
            "availability": "not_probed",
            "reason": "No executable adapter is registered",
            "probe_verified": False,
        }
        for name in PLANNED
    )
    adapters.extend(
        {
            "id": name,
            "implementation": "implemented",
            "availability": "available",
            "probe_verified": False,
            "operation": "import",
            "compatibility": "unknown_until_probe",
            "reason": (
                "Public HTTPS with required SHA-256; each import runs bounded network and native "
                "checks. Offline doctor does not contact repositories."
            ),
        }
        for name in ("HTTPS import", "Hugging Face import")
    )
    snapshot = CapabilitySnapshot(
        fingerprint="",
        checked_at=utcnow(),
        packages=versions,
        environment=environment(),
        providers=providers,
        probes=probes,
        adapters=adapters,
    )
    snapshot.fingerprint = current_fingerprint(snapshot)
    return snapshot.model_dump(mode="json")
