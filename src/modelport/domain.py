"""Framework-independent, versioned public contracts."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TensorSpec(Contract):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    dtype: Literal["float32", "float64", "float16", "int64", "int32", "int8", "uint8", "bool"]
    dimensions: list[int | str | None] = Field(max_length=8)
    shape: list[int] | None = None
    bounds: dict[str, tuple[int, int]] = Field(default_factory=dict)
    value_range: tuple[float, float] | None = None
    class_axis: int | None = None

    @property
    def rank(self) -> int:
        return len(self.dimensions)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if any(isinstance(d, int) and (d < 1 or d > 1_000_000) for d in self.dimensions):
            raise ValueError("Dimensions must be between 1 and 1,000,000")
        if self.shape is not None and (
            len(self.shape) != self.rank or any(d < 1 or d > 1_000_000 for d in self.shape)
        ):
            raise ValueError("Concrete shape must match rank with positive bounded dimensions")
        if any(lo < 1 or hi < lo or hi > 1_000_000 for lo, hi in self.bounds.values()):
            raise ValueError("Invalid symbolic dimension bounds")
        if self.class_axis is not None and not -self.rank <= self.class_axis < self.rank:
            raise ValueError("Invalid class axis")
        return self


class ArchitectureConfig(Contract):
    features: int = Field(default=64, ge=4, le=1024)
    hidden: int = Field(default=128, ge=4, le=1024)
    classes: int = Field(default=10, ge=2, le=1000)


class ModelBundleSpec(Contract):
    schema_version: Literal[1] = 1
    architecture: Literal["vision-mlp", "dual-input"]
    architecture_version: Literal["1"] = "1"
    config: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    weights: Literal["weights.safetensors"] = "weights.safetensors"
    inputs: list[TensorSpec] = Field(min_length=1, max_length=8)
    outputs: list[TensorSpec] = Field(min_length=1, max_length=8)
    expected_digests: dict[str, str] = Field(default_factory=dict)
    task: str | None = Field(default=None, max_length=256)
    preprocessing: Literal["identity"] = "identity"


class ModelState(Contract):
    format: Literal["pytorch-bundle", "onnx", "safetensors"]
    precision: Literal["fp32", "dynamic-int8", "static-int8", "fp16", "unknown"] = "unknown"
    optimized: bool = False
    architecture: str | None = None
    architecture_version: str | None = None
    opsets: dict[str, int] = Field(default_factory=dict)
    domains: list[str] = Field(default_factory=list)
    runtime: str | None = None
    provider: str | None = None


class ModelDescriptor(Contract):
    schema_version: Literal[1] = 1
    name: str
    state: ModelState
    inputs: list[TensorSpec] = Field(default_factory=list)
    outputs: list[TensorSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    runtime_compatibility: Literal["compatible", "blocked", "unknown_until_probe"]


class ArtifactRef(Contract):
    artifact_id: str
    digest: str


class ArtifactFile(Contract):
    path: str
    size_bytes: int
    sha256: str


class ArtifactBundle(ArtifactRef):
    files: list[ArtifactFile]
    descriptor: ModelDescriptor
    manifest: dict[str, Any]
    validation_state: Literal["validated", "failed", "unverified"] = "unverified"
    created_at: str = Field(default_factory=utcnow)


class PreflightReport(Contract):
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class CapabilitySnapshot(Contract):
    schema_version: Literal[1] = 1
    fingerprint: str
    checked_at: str
    packages: dict[str, str | None]
    environment: dict[str, Any]
    providers: list[str]
    probes: dict[str, Any]
    adapters: list[dict[str, Any]]


class ConversionRequest(Contract):
    source_id: str
    target: Literal["onnx"] = "onnx"
    precision: Literal["fp32", "dynamic-int8", "static-int8", "fp16"] = "fp32"
    optimize: bool = False
    device: Literal["cpu"] = "cpu"
    skip_validation: bool = False
    policy: (
        Literal["fp32-default", "int8-synthetic-v1", "int8-calibrated-v1", "fp16-default"] | None
    ) = None
    calibration_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    validation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    calibration_batch_size: int = Field(default=4, ge=1, le=64)
    batch_sizes: list[int] = Field(default_factory=lambda: [2, 4], min_length=1, max_length=8)
    seed: int = Field(default=2027, ge=0, le=2**32 - 1)


class ConversionStep(Contract):
    adapter_id: str
    version: str = "1"
    source: ModelState
    target: ModelState
    options: dict[str, Any]


class TensorDataset(Contract):
    dataset_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    logical_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    purpose: Literal["calibration", "validation"]
    sample_count: int = Field(ge=1, le=1024)
    sample_digests: list[str] = Field(min_length=1, max_length=1024)
    tensors: list[TensorSpec] = Field(min_length=1, max_length=8)
    size_bytes: int = Field(ge=1)
    created_at: str = Field(default_factory=utcnow)


class RemoteFile(Contract):
    url: str = Field(min_length=8, max_length=8192)
    path: str = Field(min_length=1, max_length=240)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class HTTPSImport(Contract):
    files: list[RemoteFile] = Field(min_length=1, max_length=64)


class HubFile(Contract):
    path: str = Field(min_length=1, max_length=240)
    destination: str | None = Field(default=None, min_length=1, max_length=240)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class HubImport(Contract):
    repository: str = Field(pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$", max_length=180)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    files: list[HubFile] = Field(min_length=1, max_length=64)


class ConversionPlan(Contract):
    schema_version: Literal[1] = 1
    plan_id: str
    request: ConversionRequest
    source_digest: str
    capability_fingerprint: str
    capability_snapshot: CapabilitySnapshot
    steps: list[ConversionStep]
    preflight: PreflightReport
    rejected_alternatives: list[str]
    datasets: dict[str, TensorDataset] = Field(default_factory=dict)


class ValidationPolicy(Contract):
    name: str = "fp32-default"
    version: str = "1"
    atol: float = Field(default=1e-5, ge=0, le=1)
    rtol: float = Field(default=1e-4, ge=0, le=1)
    relative_floor: float = Field(default=1e-8, gt=0)
    gate: Literal["allclose", "normalized-l2"] = "allclose"
    max_normalized_l2: float = Field(default=0.05, gt=0, le=1)


class ValidationReport(Contract):
    schema_version: Literal[1] = 1
    report_id: str
    created_at: str = Field(default_factory=utcnow)
    source_id: str
    target_id: str
    state: Literal["validated", "failed"]
    policy: ValidationPolicy
    dataset: dict[str, Any]
    output_mapping: dict[str, str]
    outputs: list[dict[str, Any]]
    failures: list[str]
    runtimes: dict[str, Any]
    metric_version: str = "1"
    evidence_scope: str = "synthetic numerical agreement; no task accuracy claim"
    evidence_identity: str | None = None

    def require_passed(self) -> None:
        if self.state != "validated":
            from modelport.errors import ModelPortError

            raise ModelPortError("VALIDATION_FAILED", "; ".join(self.failures))


class BenchmarkConfig(Contract):
    batch_size: int = Field(default=4, ge=1, le=64)
    warmup: int = Field(default=5, ge=0, le=100)
    iterations: int = Field(default=30, ge=2, le=1000)
    repetitions: int = Field(default=1, ge=1, le=10)
    threads: int = Field(default=1, ge=1, le=8)
    seed: int = Field(default=909, ge=0, le=2**32 - 1)


class BenchmarkReport(Contract):
    schema_version: Literal[1] = 1
    report_id: str
    created_at: str = Field(default_factory=utcnow)
    artifact_id: str
    config: BenchmarkConfig
    measurements: dict[str, Any]
    environment: dict[str, Any]
    runtime: dict[str, Any]
    validation_state: str
    artifact_bytes: int
    warnings: list[str]


JobState = Literal[
    "queued",
    "running",
    "cancel_requested",
    "succeeded",
    "failed",
    "cancelled",
    "timed_out",
    "interrupted",
]
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out", "interrupted"})


class JobRecord(Contract):
    job_id: str
    kind: str
    request: dict[str, Any]
    state: JobState
    stage: str
    attempt: int
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: str
    updated_at: str
    timeout_seconds: int


class JobEvent(Contract):
    job_id: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    created_at: str


class InferenceResult(Contract):
    outputs: dict[str, Any]
    runtime: dict[str, Any]
