"""Deterministic bounded capability graph; no executable edges for planned adapters."""

import heapq
from typing import Any

from modelport.domain import (
    ArtifactBundle,
    CapabilitySnapshot,
    ConversionPlan,
    ConversionRequest,
    ConversionStep,
    ModelState,
    PreflightReport,
    TensorDataset,
)
from modelport.errors import ModelPortError
from modelport.security import identity

ADAPTERS: list[dict[str, Any]] = [
    {
        "id": "torch-onnx",
        "version": "1",
        "implementation": "implemented",
        "source": "registered fp32 PyTorch bundle",
        "target": "fp32 ONNX",
        "packages": {"torch": "==2.13.0+cpu", "onnx": ">=1.20,<2", "onnxscript": ">=0.5,<1"},
        "options": {
            "opset": 18,
            "exporter": "dynamo",
            "fallback": False,
            "exporter_optimize": True,
        },
        "lossy": False,
        "requires": ["trusted architecture v1", "example inputs", "CPU reference runtime"],
        "restrictions": ["vision-mlp or dual-input", "batch 1..64", "float32", "CPU"],
        "entrypoint": "modelport.native:export_onnx",
        "verification": ["ONNX checker", "ORT execution", "original-source numerical validation"],
    },
    {
        "id": "onnx-optimize",
        "version": "1",
        "implementation": "implemented",
        "source": "unoptimized fp32 ONNX",
        "target": "basic-optimized fp32 ONNX",
        "packages": {"onnxruntime": ">=1.23,<2"},
        "options": {"level": "ORT_ENABLE_BASIC"},
        "lossy": False,
        "requires": ["CPU runtime"],
        "restrictions": ["standard domains", "FP32"],
        "entrypoint": "modelport.native:optimize_onnx",
        "verification": [
            "graph-change/no-op evidence",
            "ONNX checker",
            "ORT execution",
            "original-source validation",
        ],
    },
    {
        "id": "onnx-dynamic-int8",
        "version": "1",
        "implementation": "implemented",
        "source": "fp32 ONNX",
        "target": "dynamic-int8 ONNX",
        "packages": {"onnxruntime": ">=1.23,<2"},
        "options": {
            "weight_type": "QInt8",
            "per_channel": False,
            "ops": ["MatMul", "Gemm"],
            "skip_symbolic_shape": True,
        },
        "lossy": True,
        "requires": ["constant-weight eligible operators", "reference runtime"],
        "restrictions": ["standard domains", "CPU", "no static calibration"],
        "entrypoint": "modelport.native:quantize_onnx",
        "verification": [
            "changed INT8 weights and integer operators",
            "ONNX checker",
            "ORT execution",
            "original-source validation",
        ],
    },
]
ADAPTERS[2]["options"]["reduce_range"] = True
ADAPTERS.extend(
    [
        {
            "id": "onnx-static-int8",
            "version": "1",
            "implementation": "implemented",
            "source": "fp32 ONNX",
            "target": "static-int8 QDQ ONNX",
            "packages": {"onnxruntime": ">=1.23,<2"},
            "options": {
                "format": "QDQ",
                "activation_type": "QUInt8",
                "weight_type": "QInt8",
                "method": "MinMax",
                "reduce_range": True,
                "per_channel": False,
            },
            "lossy": True,
            "requires": [
                "separate calibration and validation datasets",
                "constant-weight MatMul/Gemm",
            ],
            "restrictions": [
                "standard domains",
                "CPU",
                "no identical calibration/validation samples",
            ],
            "entrypoint": "modelport.native:quantize_static_onnx",
            "verification": [
                "QDQ nodes and INT8 weights",
                "ONNX checker",
                "CPU execution",
                "held-out original-source validation",
            ],
        },
        {
            "id": "onnx-fp16",
            "version": "1",
            "implementation": "implemented",
            "source": "fp32 ONNX",
            "target": "fp16 ONNX with FP32 input/output",
            "packages": {"onnxruntime": ">=1.23,<2"},
            "options": {
                "keep_io_types": True,
                "min_positive_val": 5.96e-8,
                "max_finite_val": 65504.0,
            },
            "lossy": True,
            "requires": ["FP32 reference runtime", "runtime execution probe"],
            "restrictions": ["no control-flow subgraphs", "CPU may promote arithmetic to FP32"],
            "entrypoint": "modelport.native:float16_onnx",
            "verification": [
                "FP16 weight tensors",
                "ONNX checker",
                "CPU execution",
                "original-source validation",
            ],
        },
    ]
)
PLANNED = [
    "OpenVINO",
    "TensorRT",
    "Core ML",
    "TensorFlow/LiteRT",
    "GGUF",
]


def successors(state: ModelState, snapshot: CapabilitySnapshot):
    available = {
        a["id"]
        for a in snapshot.adapters
        if a.get("availability") == "available" and a.get("probe_verified")
    }
    if (
        state.format == "pytorch-bundle"
        and state.architecture in {"vision-mlp", "dual-input"}
        and "torch-onnx" in available
    ):
        target = state.model_copy(
            update={
                "format": "onnx",
                "opsets": {"": 18},
                "domains": [""],
                "runtime": "onnxruntime",
                "provider": "CPUExecutionProvider",
            }
        )
        yield ADAPTERS[0], target
    if state.format == "onnx" and state.precision == "fp32":
        if not state.optimized and "onnx-optimize" in available:
            yield ADAPTERS[1], state.model_copy(update={"optimized": True})
        if "onnx-dynamic-int8" in available:
            yield ADAPTERS[2], state.model_copy(update={"precision": "dynamic-int8"})
        if "onnx-static-int8" in available:
            yield ADAPTERS[3], state.model_copy(update={"precision": "static-int8"})
        if "onnx-fp16" in available:
            yield ADAPTERS[4], state.model_copy(update={"precision": "fp16"})


def build_plan(
    source: ArtifactBundle,
    request: ConversionRequest,
    snapshot: CapabilitySnapshot,
    datasets: dict[str, TensorDataset] | None = None,
) -> ConversionPlan:
    datasets = datasets or {}
    if request.precision == "static-int8":
        from modelport.datasets import require_independent

        if not request.calibration_id or not request.validation_id or request.skip_validation:
            raise ModelPortError(
                "CALIBRATION_REQUIRED",
                "Static INT8 requires registered calibration and held-out validation datasets, "
                "with validation enabled",
            )
        if set(datasets) != {"calibration", "validation"}:
            raise ModelPortError(
                "CALIBRATION_REQUIRED", "Register the requested tensor datasets first"
            )
        require_independent(datasets["calibration"], datasets["validation"])
    elif request.calibration_id:
        raise ModelPortError("INVALID_CONFIG", "Calibration applies only to static INT8")
    if source.descriptor.blockers:
        raise ModelPortError("INCOMPATIBLE_MODEL", "; ".join(source.descriptor.blockers))
    if source.validation_state == "failed":
        raise ModelPortError(
            "VALIDATION_FAILED", "Known-failed artifacts cannot be conversion sources"
        )
    if any(batch < 1 or batch > 64 for batch in request.batch_sizes):
        raise ModelPortError("INVALID_SHAPE", "Validation batch sizes must be within 1..64")
    operators = source.descriptor.metadata.get("operators", {})
    if (
        request.precision in {"dynamic-int8", "static-int8"}
        and source.descriptor.state.format == "onnx"
        and not any(op in operators for op in ("MatMul", "Gemm"))
    ):
        raise ModelPortError(
            "NO_ELIGIBLE_OPERATORS", "This graph has no MatMul/Gemm nodes for dynamic INT8"
        )
    heap: list[tuple[int, int, str, ModelState, list[ConversionStep]]] = []
    initial = source.descriptor.state
    heapq.heappush(heap, (0, 0, "", initial, []))
    seen: set[str] = set()
    selected: list[ConversionStep] | None = None
    while heap and len(seen) < 32:
        loss, depth, order, state, steps = heapq.heappop(heap)
        key = identity(state.model_dump())
        if key in seen:
            continue
        seen.add(key)
        if (
            state.format == request.target
            and state.precision == request.precision
            and state.optimized == request.optimize
        ):
            selected = steps
            break
        if depth >= 3:
            continue
        for adapter, target in successors(state, snapshot):
            step = ConversionStep(
                adapter_id=adapter["id"], source=state, target=target, options=adapter["options"]
            )
            heapq.heappush(
                heap,
                (
                    loss + int(adapter["lossy"]),
                    depth + 1,
                    order + adapter["id"],
                    target,
                    [*steps, step],
                ),
            )
    if selected is None:
        raise ModelPortError(
            "NO_ROUTE",
            (
                "No verified route satisfies the requested state. Run doctor or choose an"
                " available CPU target"
            ),
        )
    if not selected:
        raise ModelPortError(
            "NO_STATE_CHANGE",
            "The artifact already has the requested state; select a meaningful transformation",
        )
    preflight = PreflightReport(
        checked=[
            "immutable source identity",
            "trusted implemented adapters",
            "environment probe",
            "CPU requested",
            "bounded graph search",
        ],
        warnings=[
            "Synthetic numerical agreement does not measure task accuracy",
            *(
                ["INT8 is lossy; the synthetic preset is not a universal task policy"]
                if request.precision in {"dynamic-int8", "static-int8"}
                else []
            ),
            *(
                ["Explicit validation skip: output remains unverified"]
                if request.skip_validation
                else []
            ),
        ],
        unresolved=[
            "Model-specific export/operator execution requires the bounded worker probe",
            "Constant-weight eligibility checked during quantization",
        ]
        if request.precision in {"dynamic-int8", "static-int8"}
        else ["Model-specific export and runtime execution checked in worker"],
    )
    payload = {
        "request": request.model_dump(mode="json"),
        "source": source.digest,
        "capabilities": snapshot.fingerprint,
        "steps": [s.model_dump(mode="json") for s in selected],
        "datasets": {name: item.model_dump(mode="json") for name, item in datasets.items()},
    }
    return ConversionPlan(
        plan_id=identity(payload),
        request=request,
        source_digest=source.digest,
        capability_fingerprint=snapshot.fingerprint,
        capability_snapshot=snapshot,
        steps=selected,
        datasets=datasets,
        preflight=preflight,
        rejected_alternatives=[
            "Planned adapters excluded",
            "Repeated optimization excluded",
            "Precision reversal unavailable",
            "GPU fallback disabled",
        ],
    )


def assert_plan_current(
    plan: ConversionPlan,
    source: ArtifactBundle,
    snapshot: CapabilitySnapshot,
    datasets: dict[str, TensorDataset] | None = None,
) -> None:
    current = build_plan(source, plan.request, snapshot, datasets)
    if (
        current.plan_id != plan.plan_id
        or current.steps != plan.steps
        or current.datasets != plan.datasets
    ):
        raise ModelPortError(
            "STALE_PLAN",
            (
                "Source, normalized options, adapter implementation or capability "
                "evidence changed; create a new plan"
            ),
            status=409,
        )
