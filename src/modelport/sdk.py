"""Local SDK: shared application services, independent of HTTP or a running server."""

import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from modelport.config import Settings
from modelport.doctor import current_fingerprint
from modelport.domain import (
    TERMINAL,
    ArtifactBundle,
    BenchmarkConfig,
    BenchmarkReport,
    CapabilitySnapshot,
    ConversionPlan,
    ConversionRequest,
    HTTPSImport,
    HubImport,
    InferenceResult,
    JobRecord,
    TensorDataset,
    ValidationReport,
)
from modelport.errors import ModelPortError
from modelport.persistence import Repository
from modelport.planning import assert_plan_current, build_plan
from modelport.processes import ChildProcess, FileLock
from modelport.security import canonical, load_npz, read_json, save_npz, write_json
from modelport.storage import ArtifactStore
from modelport.supervisor import Supervisor


class LoadedModel:
    """One caller at a time; native runtime lives in a killable process."""

    def __init__(
        self, client: "ModelPort", artifact: ArtifactBundle, preprocess=None, postprocess=None
    ):
        self.preprocess, self.postprocess = preprocess, postprocess
        self.settings = client.settings
        self.inputs, self.outputs = artifact.descriptor.inputs, artifact.descriptor.outputs
        self.work = self.settings.data_dir / "work" / ("session-" + uuid.uuid4().hex)
        self.work.mkdir(mode=0o700)
        write_json(
            self.work / "session.json",
            {
                "settings": self.settings.model_dump(mode="json"),
                "path": str(client.store.verify(artifact.artifact_id)),
                "descriptor": artifact.descriptor.model_dump(mode="json"),
            },
        )
        self.child = ChildProcess(
            ["-m", "modelport.session_runner"], self.work, self.settings.memory_limit_bytes
        )
        self.index, self.closed = 0, False
        self.lock = threading.Lock()
        try:
            self.runtime = self._await("ready.json")["runtime"]
        except BaseException:
            self.close()
            raise

    def _await(self, filename: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.settings.timeout_seconds
        path = self.work / filename
        while not path.exists():
            if self.child.process.poll() is not None:
                raise ModelPortError("WORKER_CRASHED", "Native session exited; reopen the model")
            if time.monotonic() > deadline:
                self.close()
                raise ModelPortError("TIMEOUT", "Native session request exceeded deadline")
            time.sleep(0.02)
        # Files are written through an atomic rename to avoid reading a partial response.
        response = read_json(path, self.settings.max_ipc_bytes)
        if not response["ok"]:
            error = response["error"]
            raise ModelPortError(error["code"], error["detail"])
        return response

    def predict(self, inputs: dict[str, Any]) -> InferenceResult:
        from modelport.tensors import validate_inputs

        if self.closed:
            raise ModelPortError("SESSION_CLOSED", "Open a new loaded-model context")
        if not self.lock.acquire(blocking=False):
            raise ModelPortError("SESSION_BUSY", "A loaded model accepts one prediction at a time")
        try:
            if self.preprocess is not None:
                inputs = self.preprocess(inputs)
            validate_inputs(inputs, self.inputs, self.settings.max_tensor_bytes)
            index = self.index
            save_npz(self.work / f"input-{index}.npz", inputs)
            write_json(self.work / f"request-{index}.json", {"operation": "predict"})
            try:
                self._await(f"result-{index}.json")
                outputs = load_npz(self.work / f"output-{index}.npz")
            finally:
                self.index += 1
                for stem in (
                    f"input-{index}.npz",
                    f"request-{index}.json",
                    f"result-{index}.json",
                    f"output-{index}.npz",
                ):
                    (self.work / stem).unlink(missing_ok=True)
            if self.postprocess is not None:
                outputs = self.postprocess(outputs)
            return InferenceResult(outputs=outputs, runtime=self.runtime)
        finally:
            self.lock.release()

    def warmup(self, inputs: dict[str, Any], count: int = 3) -> None:
        if not 0 <= count <= 100:
            raise ModelPortError("INVALID_CONFIG", "Warmup count must be within 0..100")
        for _ in range(count):
            self.predict(inputs)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.child.close()
            if self.work.resolve().parent == (self.settings.data_dir / "work").resolve():
                shutil.rmtree(self.work, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ModelPort:
    def __init__(self, settings: Settings, *, migrate: bool = True):
        self.settings = settings
        settings.prepare()
        self.repository = Repository(settings)
        if migrate:
            lock = FileLock(settings.data_dir / "migration.lock")
            if not lock.acquire(blocking=True):
                raise ModelPortError("DATABASE_BUSY", "Another process is migrating the database")
            try:
                self.repository.migrate()
            finally:
                lock.close()
        self.store = ArtifactStore(settings)
        self.supervisor = Supervisor(settings, self.repository)
        self.sessions: list[LoadedModel] = []

    @classmethod
    def local(cls, data_dir: str | Path | None = None, **options):
        return cls(
            Settings.model_validate(
                {**({"data_dir": data_dir} if data_dir is not None else {}), **options}
            )
        )

    def wait(self, job: JobRecord | str) -> JobRecord:
        job_id = job if isinstance(job, str) else job.job_id
        try:
            while True:
                current = self.repository.job(job_id)
                if current.state in TERMINAL:
                    if current.state != "succeeded":
                        error = current.error or {
                            "code": current.state.upper(),
                            "detail": f"Job {current.state}",
                        }
                        raise ModelPortError(
                            error["code"], error["detail"], job_id=job_id, result=current.result
                        )
                    return current
                if not self.supervisor.run_one():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.repository.cancel(job_id)
            self.supervisor.close()
            raise

    @staticmethod
    def result(job: JobRecord) -> dict[str, Any]:
        if job.result is None:
            raise ModelPortError("INCOMPLETE_JOB", "Job has no complete result")
        return job.result

    def doctor(self) -> CapabilitySnapshot:
        job = self.wait(self.repository.submit("doctor", {}))
        return CapabilitySnapshot.model_validate(self.result(job)["capabilities"])

    def capabilities(self, *, probe: bool = True) -> CapabilitySnapshot:
        snapshot = self.repository.get_setting("capabilities")
        if snapshot:
            parsed = CapabilitySnapshot.model_validate(snapshot)
            if current_fingerprint(parsed) == parsed.fingerprint:
                return parsed
        if probe:
            return self.doctor()
        raise ModelPortError(
            "CAPABILITIES_PENDING",
            "Run doctor; verified capability evidence is absent or stale",
            status=503,
        )

    def import_model(self, path: str | Path, *, wait: bool = True) -> ArtifactBundle | JobRecord:
        staging_id, name = self.store.stage_import(Path(path))
        job = self.repository.submit("import", {"staging_id": staging_id, "name": name})
        return self.repository.artifact(self.result(self.wait(job))["artifact_id"]) if wait else job

    def import_https(
        self, files: list[dict[str, Any]], *, wait: bool = True
    ) -> ArtifactBundle | JobRecord:
        from modelport.remote import validate_files

        request = HTTPSImport.model_validate({"files": files})
        validate_files(request)
        job = self.repository.submit("remote-import", {"request": request.model_dump(mode="json")})
        return self.repository.artifact(self.result(self.wait(job))["artifact_id"]) if wait else job

    def import_hub(
        self, repository: str, revision: str, files: list[dict[str, Any]], *, wait: bool = True
    ) -> ArtifactBundle | JobRecord:
        from modelport.remote import hub_request, validate_files

        request = HubImport.model_validate(
            {"repository": repository, "revision": revision, "files": files}
        )
        validate_files(hub_request(request))
        job = self.repository.submit("hub-import", {"request": request.model_dump(mode="json")})
        return self.repository.artifact(self.result(self.wait(job))["artifact_id"]) if wait else job

    def fixture(
        self, architecture: str = "vision-mlp", *, wait: bool = True
    ) -> ArtifactBundle | JobRecord:
        if architecture not in {"vision-mlp", "dual-input"}:
            raise ModelPortError("MISSING_ARCHITECTURE", "Choose vision-mlp or dual-input")
        job = self.repository.submit("fixture", {"architecture": architecture})
        return self.repository.artifact(self.result(self.wait(job))["artifact_id"]) if wait else job

    def import_dataset(
        self, path: str | Path, *, purpose: str, wait: bool = True
    ) -> TensorDataset | JobRecord:
        if purpose not in {"calibration", "validation"}:
            raise ModelPortError("DATASET_PURPOSE", "Choose calibration or validation")
        staging_id = self.stage_dataset(path)
        job = self.repository.submit("dataset", {"staging_id": staging_id, "purpose": purpose})
        return self.repository.dataset(self.result(self.wait(job))["dataset_id"]) if wait else job

    def plan_datasets(self, request: ConversionRequest) -> dict[str, TensorDataset]:
        selected: dict[str, TensorDataset] = {}
        for purpose, identifier in (
            ("calibration", request.calibration_id),
            ("validation", request.validation_id),
        ):
            if identifier:
                dataset = self.repository.dataset(identifier)
                if dataset.purpose != purpose:
                    raise ModelPortError("DATASET_PURPOSE", f"Select a {purpose} dataset")
                selected[purpose] = dataset
        return selected

    def inspect(self, artifact_id: str):
        return self.repository.artifact(artifact_id).descriptor

    def plan(
        self,
        source_id: str,
        *,
        target: str = "onnx",
        precision: str = "fp32",
        device: str = "cpu",
        **options,
    ) -> ConversionPlan:
        if device != "cpu":
            raise ModelPortError(
                "DEVICE_UNAVAILABLE",
                "Only verified CPU execution is supported; no implicit device fallback",
            )
        request = ConversionRequest.model_validate(
            {
                "source_id": source_id,
                "target": target,
                "precision": precision,
                "device": device,
                **options,
            }
        )
        return build_plan(
            self.repository.artifact(source_id),
            request,
            self.capabilities(),
            self.plan_datasets(request),
        )

    def convert(
        self, plan: ConversionPlan, *, wait: bool = True, idempotency_key: str | None = None
    ) -> ArtifactBundle | JobRecord:
        assert_plan_current(
            plan,
            self.repository.artifact(plan.request.source_id),
            self.capabilities(probe=False),
            self.plan_datasets(plan.request),
        )
        job = self.repository.submit(
            "convert", {"plan": plan.model_dump(mode="json")}, idempotency_key
        )
        return self.repository.artifact(self.result(self.wait(job))["artifact_id"]) if wait else job

    def stage_dataset(self, path: str | Path) -> str:
        source = Path(path)
        if not source.is_file() or source.stat().st_size > self.settings.max_file_bytes:
            raise ModelPortError(
                "INVALID_INPUT", "Dataset is missing or exceeds the input byte limit"
            )
        identifier = uuid.uuid4().hex
        target = self.settings.data_dir / "staging" / identifier
        target.mkdir(mode=0o700)
        # Parsing occurs only in the supervised native process.
        shutil.copyfile(source, target / "inputs.npz")
        return identifier

    def validate(
        self,
        source_id: str,
        target_id: str,
        *,
        inputs: str | Path | None = None,
        validation_id: str | None = None,
        policy: str = "fp32-default",
        wait: bool = True,
        mapping: dict[str, str] | None = None,
        batch_sizes: list[int] | None = None,
        seed: int = 2027,
        idempotency_key: str | None = None,
    ) -> ValidationReport | JobRecord:
        from modelport.validation import policy_named

        policy_named(policy)
        if inputs and validation_id:
            raise ModelPortError(
                "INVALID_CONFIG", "Choose an NPZ path or a registered validation dataset"
            )
        if validation_id and self.repository.dataset(validation_id).purpose != "validation":
            raise ModelPortError("DATASET_PURPOSE", "Choose a validation dataset")
        self.repository.artifact(source_id)
        self.repository.artifact(target_id)
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "policy": policy,
            "mapping": mapping,
            "batch_sizes": batch_sizes,
            "seed": seed,
            "validation_id": validation_id,
        }
        if inputs:
            payload["dataset_id"] = self.stage_dataset(inputs)
        job = self.repository.submit("validate", payload, idempotency_key)
        if not wait:
            return job
        try:
            done = self.wait(job)
        except ModelPortError as exc:
            if exc.code != "VALIDATION_FAILED":
                raise
            done = self.repository.job(job.job_id)
        return ValidationReport.model_validate(
            self.repository.report(self.result(done)["validation_id"])
        )

    def benchmark(
        self,
        artifact_id: str,
        config: BenchmarkConfig | None = None,
        *,
        wait: bool = True,
        idempotency_key: str | None = None,
    ) -> BenchmarkReport | JobRecord:
        source = self.repository.artifact(artifact_id)
        if source.validation_state == "failed":
            raise ModelPortError(
                "VALIDATION_FAILED", "Known-failed artifacts cannot be benchmarked"
            )
        job = self.repository.submit(
            "benchmark",
            {
                "source_id": artifact_id,
                "config": (config or BenchmarkConfig()).model_dump(mode="json"),
            },
            idempotency_key,
        )
        return (
            BenchmarkReport.model_validate(
                self.repository.report(self.result(self.wait(job))["benchmark_id"])
            )
            if wait
            else job
        )

    def load(
        self,
        artifact_id: str,
        *,
        device: str = "cpu",
        require_validated: bool = False,
        preprocess=None,
        postprocess=None,
    ) -> LoadedModel:
        if device != "cpu":
            raise ModelPortError(
                "DEVICE_UNAVAILABLE", "Only CPU is verified; no device fallback is configured"
            )
        artifact = self.repository.artifact(artifact_id)
        if artifact.validation_state == "failed" or (
            require_validated and artifact.validation_state != "validated"
        ):
            raise ModelPortError(
                "VALIDATION_REQUIRED", "Artifact does not satisfy the requested validation policy"
            )
        session = LoadedModel(self, artifact, preprocess=preprocess, postprocess=postprocess)
        self.sessions.append(session)
        return session

    def export_artifact(self, artifact_id: str, destination: Path) -> Path:
        artifact = self.repository.artifact(artifact_id)
        directory = self.store.verify(artifact_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in artifact.files:
                archive.write(directory / file.path, file.path)
            archive.writestr(
                "modelport-provenance.json", canonical(artifact.model_dump(mode="json"))
            )
            archive.writestr(
                "modelport-evidence.json", canonical(self.repository.list_reports(artifact_id))
            )
        return destination

    def close(self) -> None:
        for session in self.sessions:
            session.close()
        self.supervisor.close()
        self.repository.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
