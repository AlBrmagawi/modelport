"""Independent numerical evidence; every required output on every batch must pass."""

import uuid
from pathlib import Path
from typing import Any

from modelport.config import Settings
from modelport.domain import ModelDescriptor, ValidationPolicy, ValidationReport
from modelport.errors import ModelPortError
from modelport.security import digest_file, identity, load_npz
from modelport.tensors import synthetic


def policy_named(name: str) -> ValidationPolicy:
    if name == "fp32-default":
        return ValidationPolicy()
    if name == "int8-synthetic-v1":
        return ValidationPolicy(
            name=name, gate="normalized-l2", max_normalized_l2=0.05, atol=1e-5, rtol=1e-4
        )
    if name in {"int8-calibrated-v1", "fp16-default"}:
        return ValidationPolicy(
            name=name,
            gate="normalized-l2",
            max_normalized_l2=0.005 if name == "fp16-default" else 0.05,
        )
    raise ModelPortError("INVALID_POLICY", "Unknown versioned tolerance policy")


def compare_arrays(
    reference: Any, actual: Any, policy: ValidationPolicy, class_axis: int | None = None
) -> dict[str, Any]:
    import numpy as np

    result: dict[str, Any] = {
        "passed": False,
        "reasons": [],
        "shape": list(reference.shape),
        "source_dtype": str(reference.dtype),
        "target_dtype": str(actual.dtype),
    }
    if reference.shape != actual.shape:
        result["reasons"].append("Output shape mismatch")
        return result
    if reference.dtype != actual.dtype:
        result["reasons"].append("Output dtype mismatch")
        return result
    if reference.size == 0:
        result["reasons"].append("Empty output cannot establish numerical agreement")
        return result
    if not (np.isfinite(reference).all() and np.isfinite(actual).all()):
        result["reasons"].append("NaN or infinity in output")
        return result
    if reference.dtype.kind in "iub":
        result["passed"] = bool(np.array_equal(reference, actual))
        result["exact_agreement"] = result["passed"]
        if not result["passed"]:
            result["reasons"].append("Nonfloating outputs must match exactly")
        return result
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    magnitude = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if magnitude > float(np.sqrt(np.finfo(np.float64).max / a.size)) / 4:
        result["reasons"].append("Output magnitude exceeds the safe float64 accumulation budget")
        result["normalized_l2"] = None
        result["cosine_similarity"] = None
        return result
    absolute = np.abs(a - b)
    relative = absolute / np.maximum(np.abs(a), policy.relative_floor)
    norm_a, norm_b = float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel()))
    delta = float(np.linalg.norm((a - b).ravel()))
    normalized_l2 = delta / max(norm_a, policy.relative_floor)
    cosine = float(np.dot(a.ravel(), b.ravel()) / norm_a / norm_b) if norm_a and norm_b else None
    result.update(
        {
            "max_absolute_error": float(absolute.max()),
            "mean_absolute_error": float(absolute.mean()),
            "max_relative_error": float(relative.max()),
            "mean_relative_error": float(relative.mean()),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "normalized_l2": normalized_l2,
            "cosine_similarity": min(1.0, max(-1.0, cosine)) if cosine is not None else None,
            "cosine_note": None if cosine is not None else "Undefined for a zero vector",
            "elements": int(a.size),
        }
    )
    if class_axis is not None:
        result["top1_agreement"] = float(
            np.mean(np.argmax(a, axis=class_axis) == np.argmax(b, axis=class_axis))
        )
    result["passed"] = (
        bool(np.allclose(a, b, atol=policy.atol, rtol=policy.rtol))
        if policy.gate == "allclose"
        else normalized_l2 <= policy.max_normalized_l2
    )
    if not result["passed"]:
        result["reasons"].append(f"Required {policy.gate} gate failed")
    return result


def validate_models(
    source_path: Path,
    source: ModelDescriptor,
    source_id: str,
    target_path: Path,
    target: ModelDescriptor,
    target_id: str,
    settings: Settings,
    policy: ValidationPolicy,
    batches: list[int] | None = None,
    seed: int = 2027,
    inputs_path: Path | None = None,
    mapping: dict[str, str] | None = None,
) -> ValidationReport:
    import numpy as np

    from modelport.native import NativeModel

    expected = {s.name for s in source.outputs}
    mapping = mapping or {name: name for name in sorted(expected)}
    if (
        set(mapping) != expected
        or len(set(mapping.values())) != len(mapping)
        or set(mapping.values()) != {s.name for s in target.outputs}
    ):
        raise ModelPortError(
            "OUTPUT_MAPPING", "Output mapping must be explicit, complete and one-to-one"
        )
    batch_sizes = batches or [2, 4]
    data = (
        [load_npz(inputs_path)]
        if inputs_path
        else [synthetic(source.inputs, batch, seed + i) for i, batch in enumerate(batch_sizes)]
    )
    dataset = {
        "kind": "npz" if inputs_path else "synthetic-normal-v1",
        "seed": None if inputs_path else seed,
        "sha256": digest_file(inputs_path)
        if inputs_path
        else identity(
            {
                "generator": "numpy-default_rng-normal-v1",
                "numpy": np.__version__,
                "seed": seed,
                "batches": batch_sizes,
                "specs": [s.model_dump() for s in source.inputs],
            }
        ),
        "batch_count": len(data),
        "sample_count": sum(next(iter(x.values())).shape[0] for x in data),
        "shapes": [{n: list(v.shape) for n, v in batch.items()} for batch in data],
        "preprocessing": "identity",
    }
    reference, candidate = (
        NativeModel(source_path, source, settings),
        NativeModel(target_path, target, settings),
    )
    metrics, failures = [], []
    try:
        for index, inputs in enumerate(data):
            a = reference.predict({n: v.copy() for n, v in inputs.items()})
            b = candidate.predict({n: v.copy() for n, v in inputs.items()})
            repeat_a = reference.predict({n: v.copy() for n, v in inputs.items()})
            repeat_b = candidate.predict({n: v.copy() for n, v in inputs.items()})
            for output in source.outputs:
                name = output.name
                row = {
                    "output": name,
                    "target_output": mapping[name],
                    "batch_index": index,
                    **compare_arrays(a[name], b[mapping[name]], policy, output.class_axis),
                }
                row["repeat_deterministic"] = bool(
                    np.array_equal(a[name], repeat_a[name])
                    and np.array_equal(b[mapping[name]], repeat_b[mapping[name]])
                )
                if not row["repeat_deterministic"]:
                    row["passed"] = False
                    row["reasons"].append("Runtime nondeterminism observed in exact repeat check")
                if not row["passed"]:
                    failures.append(f"{name}, batch {index}: " + "; ".join(row["reasons"]))
                metrics.append(row)
        report = ValidationReport(
            report_id=uuid.uuid4().hex,
            source_id=source_id,
            target_id=target_id,
            state="failed" if failures else "validated",
            policy=policy,
            dataset=dataset,
            output_mapping=mapping,
            outputs=metrics,
            failures=failures,
            runtimes={"source": reference.info, "target": candidate.info},
            evidence_scope="provided tensor dataset numerical agreement; no task labels"
            if inputs_path
            else "synthetic numerical agreement; no task accuracy claim",
        )
        report.evidence_identity = identity(
            {
                "source": source_id,
                "target": target_id,
                "dataset": dataset,
                "mapping": mapping,
                "policy": policy.model_dump(),
                "runtimes": report.runtimes,
                "metric_version": report.metric_version,
            }
        )
        return report
    finally:
        reference.close()
        candidate.close()
