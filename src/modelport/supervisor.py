import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from modelport.config import Settings
from modelport.domain import ArtifactBundle, ArtifactFile, TensorDataset, utcnow
from modelport.errors import ModelPortError
from modelport.persistence import Repository, attempts, checkpoints, datasets, jobs, logs
from modelport.processes import ChildProcess, FileLock
from modelport.security import confined, digest_file, read_json, write_json
from modelport.storage import ArtifactStore


class Supervisor:
    def __init__(self, settings: Settings, repository: Repository):
        self.settings, self.repository = settings, repository
        self.store = ArtifactStore(settings)
        self.stop_event = threading.Event()
        self.stop_file: Path | None = None
        self.child: ChildProcess | None = None
        self.lock = FileLock(settings.data_dir / "worker.lock")
        self.worker_id = uuid.uuid4().hex

    def artifact_input(self, artifact_id: str) -> dict[str, Any]:
        artifact = self.repository.artifact(artifact_id)
        return {**artifact.model_dump(mode="json"), "path": str(self.store.verify(artifact_id))}

    def prepare_payload(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = dict(job["request"])
        if job["kind"] in {"import", "dataset"}:
            identifier = payload["staging_id"]
            if not re.fullmatch(r"[a-f0-9]{32}", identifier):
                raise ModelPortError("UNSAFE_PATH", "Invalid staging identifier")
            payload["staging_path"] = str(self.settings.data_dir / "staging" / identifier)
        elif job["kind"] == "convert":
            source_id = payload["plan"]["request"]["source_id"]
            source = self.artifact_input(source_id)
            payload["source"] = source
            payload["reference"] = self.artifact_input(
                source["manifest"].get("reference_id", source_id)
            )
            payload["checkpoints"] = {}
            for purpose in ("calibration", "validation"):
                identifier = payload["plan"]["request"].get(purpose + "_id")
                if identifier:
                    payload[purpose + "_path"] = str(self.dataset_path(identifier, purpose))
            with self.repository.engine.connect() as connection:
                saved = (
                    connection.execute(
                        select(checkpoints)
                        .where(
                            checkpoints.c.job_id == job["job_id"],
                            checkpoints.c.plan_id == payload["plan"]["plan_id"],
                        )
                        .order_by(checkpoints.c.step_index)
                    )
                    .mappings()
                    .all()
                )
            for item in saved:
                path = self.settings.data_dir / "checkpoints" / item["digest"]
                if path.is_dir() and self.store.inventory(path)[0] == item["digest"]:
                    payload["checkpoints"][str(item["step_index"])] = {
                        **item["data"],
                        "path": str(path),
                    }
        elif job["kind"] in {"validate", "benchmark", "predict"}:
            payload["source"] = self.artifact_input(payload["source_id"])
            if job["kind"] == "validate":
                payload["target"] = self.artifact_input(payload["target_id"])
            if payload.get("dataset_id"):
                identifier = payload["dataset_id"]
                if not re.fullmatch(r"[a-f0-9]{32}", identifier):
                    raise ModelPortError("UNSAFE_PATH", "Invalid dataset identifier")
                payload["inputs_path"] = str(
                    self.settings.data_dir / "staging" / identifier / "inputs.npz"
                )
            if payload.get("validation_id"):
                payload["inputs_path"] = str(self.dataset_path(payload["validation_id"]))
        if (
            job["kind"] in {"convert", "predict", "benchmark"}
            and payload["source"]["validation_state"] == "failed"
        ):
            raise ModelPortError(
                "VALIDATION_FAILED", "Known-failed artifacts are rejected by default"
            )
        return payload

    def dataset_path(self, identifier: str, purpose: str = "validation") -> Path:
        dataset = self.repository.dataset(identifier)
        if dataset.purpose != purpose:
            raise ModelPortError("DATASET_PURPOSE", f"A {purpose} dataset is required")
        path = confined(self.settings.data_dir / "datasets", dataset.sha256 + "/inputs.npz")
        if digest_file(path) != dataset.sha256:
            raise ModelPortError("INTEGRITY_ERROR", "Dataset bytes no longer match their identity")
        return path

    def tick_ready(self) -> None:
        self.repository.set_setting("worker", {"id": self.worker_id, "heartbeat": time.time()})

    def run_one(self) -> bool:
        if not self.lock.acquire():
            return False
        try:
            self.tick_ready()
            self.recover_checkpoints()
            self.repository.recover()
            job = self.repository.claim()
            if job is None:
                return False
            self.execute(job)
            return True
        finally:
            self.lock.close()

    def execute(self, job: dict[str, Any]) -> None:
        repo, settings = self.repository, self.settings
        work = settings.data_dir / "work" / f"{job['job_id']}-{job['token']}"
        work.mkdir(mode=0o700)
        state, result, error, exit_code = "failed", None, None, None
        try:
            payload = self.prepare_payload(job)
            write_json(
                work / "request.json",
                {
                    "kind": job["kind"],
                    "payload": payload,
                    "settings": settings.model_dump(mode="json"),
                    "timeout_seconds": job["timeout_seconds"],
                },
            )
            self.child = ChildProcess(
                ["-m", "modelport.runner"],
                work,
                settings.memory_limit_bytes,
                settings.threads,
                settings.max_log_bytes,
            )
            with repo.transaction() as connection:
                connection.execute(
                    update(attempts)
                    .where(attempts.c.job_id == job["job_id"], attempts.c.token == job["token"])
                    .values(pid=self.child.process.pid)
                )
            deadline, cursor = time.monotonic() + job["timeout_seconds"], 0
            while self.child.process.poll() is None:
                if self.stop_file and self.stop_file.exists():
                    self.stop_event.set()
                if self.stop_event.is_set():
                    state = "interrupted"
                    self.child.kill()
                    break
                if not repo.heartbeat(job["job_id"], job["token"]):
                    state = "cancelled"
                    self.child.kill()
                    break
                if time.monotonic() >= deadline:
                    state = "timed_out"
                    error = ModelPortError(
                        "TIMEOUT", "Job exceeded its configured wall-clock deadline"
                    ).problem()
                    self.child.kill()
                    break
                event_file = work / "events.jsonl"
                if event_file.exists():
                    if event_file.stat().st_size > settings.max_log_bytes:
                        raise ModelPortError(
                            "RESOURCE_EXHAUSTED", "Worker event output exceeded budget"
                        )
                    with event_file.open(encoding="utf-8") as stream:
                        stream.seek(cursor)
                        while line := stream.readline():
                            if not line.endswith("\n"):
                                break
                            event = json.loads(line)
                            repo.progress(
                                job["job_id"], job["token"], event["stage"], event["payload"]
                            )
                            cursor = stream.tell()
                self.tick_ready()
                self.stop_event.wait(0.15)
            exit_code = self.child.process.poll()
            if state not in {"cancelled", "timed_out", "interrupted"}:
                response_path = work / "result.json"
                if exit_code != 0 or not response_path.exists():
                    raise ModelPortError(
                        "WORKER_CRASHED",
                        "Native worker exited without a complete result; no output was published",
                    )
                response = read_json(response_path, settings.max_ipc_bytes)
                if response["ok"]:
                    result = self.publish(job, work, response["result"])
                    return
                error = response["error"]
                state = "timed_out" if error["code"] == "TIMEOUT" else "failed"
        except KeyboardInterrupt:
            state = "interrupted"
            error = ModelPortError("INTERRUPTED", "Operator interrupted the local worker").problem()
            raise
        except ModelPortError as exc:
            error = exc.problem()
        except Exception as exc:
            error = ModelPortError(
                "SUPERVISOR_ERROR",
                f"{type(exc).__name__} in worker supervision; no partial output published",
            ).problem()
        finally:
            if self.child:
                self.child.close()
                text = self.child.output.decode("utf-8", errors="replace")
                text = text.replace(str(settings.data_dir), "[data]").replace(
                    str(Path.home()), "[home]"
                )
                text = re.sub(
                    r"(?i)(bearer\s+|token[=:]\s*|password[=:]\s*)\S+", r"\1[redacted]", text
                )
                text = re.sub(r"https?://\S+", "[url]", text)
                if self.child.truncated:
                    text += "\n[log truncated at byte budget]"
                with repo.transaction() as connection:
                    from sqlalchemy.dialects.sqlite import insert as upsert

                    connection.execute(
                        upsert(logs)
                        .values(job_id=job["job_id"], text=text)
                        .on_conflict_do_update(index_elements=["job_id"], set_={"text": text})
                    )
                self.child = None
            if job["kind"] == "convert":
                self.save_checkpoints(job, work)
            with repo.transaction() as connection:
                repo.finish(
                    connection, job["job_id"], job["token"], state, result, error, exit_code
                )
            # Work roots are generated internally and verified before recursive deletion.
            if work.resolve().parent == (settings.data_dir / "work").resolve():
                shutil.rmtree(work, ignore_errors=True)

    def save_checkpoints(self, job: dict[str, Any], work: Path) -> None:
        from sqlalchemy.dialects.sqlite import insert as upsert

        plan = job["request"]["plan"]
        for index in range(max(0, len(plan["steps"]) - 1)):
            record = work / f"checkpoint-{index}.json"
            if not record.is_file():
                continue
            try:
                data = read_json(record, self.settings.max_ipc_bytes)
            except (ModelPortError, OSError):
                continue  # Incomplete checkpoint metadata is never evidence of a completed step.
            if data.get("directory") != f"step-{index}" or data.get("plan_id") != plan["plan_id"]:
                continue
            path = work / f"step-{index}"
            if not path.is_dir():
                continue
            try:
                digest, files = self.store.inventory(path)
            except (ModelPortError, OSError):
                continue
            if digest != data["digest"] or files != data["files"]:
                continue
            with self.repository.transaction() as connection:
                owner = connection.scalar(
                    select(jobs.c.token).where(jobs.c.job_id == job["job_id"])
                )
                if owner != job["token"]:
                    return
                destination = self.settings.data_dir / "checkpoints" / digest
                if destination.exists():
                    if self.store.inventory(destination)[0] != digest:
                        continue
                else:
                    os.replace(path, destination)
                connection.execute(
                    upsert(checkpoints)
                    .values(
                        job_id=job["job_id"],
                        step_index=index,
                        plan_id=plan["plan_id"],
                        digest=digest,
                        data=data,
                        created_at=utcnow(),
                    )
                    .on_conflict_do_update(
                        index_elements=["job_id", "step_index"],
                        set_={"digest": digest, "data": data, "plan_id": plan["plan_id"]},
                    )
                )

    def recover_checkpoints(self) -> None:
        # The OS worker lock excludes another live supervisor. Only expired leases
        # are harvested; complete files are rehashed before reuse on a new attempt.
        with self.repository.engine.connect() as connection:
            stale = (
                connection.execute(
                    select(jobs).where(
                        jobs.c.kind == "convert",
                        jobs.c.state.in_(["running", "cancel_requested"]),
                        jobs.c.heartbeat < time.time() - 15,
                    )
                )
                .mappings()
                .all()
            )
        for job in stale:
            work = self.settings.data_dir / "work" / f"{job['job_id']}-{job['token']}"
            if work.is_dir():
                self.save_checkpoints(dict(job), work)

    def publish(self, job: dict[str, Any], work: Path, response: dict[str, Any]) -> dict[str, Any]:
        repo = self.repository
        result: dict[str, Any] = {}
        # Content verification is outside the transaction; ownership checked immediately
        # before filesystem publication and DB visibility in a short fenced transaction.
        candidate = response.get("candidate")
        dataset = (
            TensorDataset.model_validate(response["dataset"]) if "dataset" in response else None
        )
        if dataset and digest_file(work / "dataset" / "inputs.npz") != dataset.sha256:
            raise ModelPortError("INTEGRITY_ERROR", "Dataset changed before publication")
        if candidate:
            if candidate["directory"] != "candidate":
                raise ModelPortError("UNSAFE_PATH", "Worker proposed an invalid publication path")
            digest, files = self.store.inventory(work / "candidate")
            if digest != candidate["digest"] or files != candidate["files"]:
                raise ModelPortError("INTEGRITY_ERROR", "Worker output changed before publication")
        with repo.transaction() as connection:
            if not repo.owns(connection, job["job_id"], job["token"]):
                raise ModelPortError("STALE_ATTEMPT", "Attempt ownership changed; result discarded")
            if candidate:
                artifact_id, files = self.store.publish(work / "candidate")
                artifact = ArtifactBundle(
                    artifact_id=artifact_id,
                    digest=artifact_id,
                    files=[ArtifactFile.model_validate(file) for file in files],
                    descriptor=candidate["descriptor"],
                    manifest=candidate["manifest"],
                )
                repo.save_artifact(connection, artifact)
                result["artifact_id"] = artifact_id
            if dataset:
                destination = self.settings.data_dir / "datasets" / dataset.sha256
                if destination.exists():
                    if digest_file(destination / "inputs.npz") != dataset.sha256:
                        raise ModelPortError("INTEGRITY_ERROR", "Stored dataset was modified")
                else:
                    os.replace(work / "dataset", destination)
                from sqlalchemy.dialects.sqlite import insert as upsert

                connection.execute(
                    upsert(datasets)
                    .values(
                        dataset_id=dataset.dataset_id,
                        data=dataset.model_dump(mode="json"),
                        created_at=dataset.created_at,
                    )
                    .on_conflict_do_nothing(index_elements=["dataset_id"])
                )
                result["dataset_id"] = dataset.dataset_id
            for name in ("validation", "benchmark"):
                if name in response:
                    report = response[name]
                    artifact_id = (
                        report["target_id"] if name == "validation" else report["artifact_id"]
                    )
                    repo.save_report(connection, name, report, artifact_id)
                    result[name + "_id"] = report["report_id"]
            if "capabilities" in response:
                repo.set_setting_in(connection, "capabilities", response["capabilities"])
                result["capabilities"] = response["capabilities"]
            if "outputs" in response:
                result.update(response)
            failed = response.get("failed_validation", False)
            repo.finish(
                connection,
                job["job_id"],
                job["token"],
                "failed" if failed else "succeeded",
                result,
                ModelPortError(
                    "VALIDATION_FAILED",
                    "Required numerical validation failed; diagnostic artifact and report retained",
                ).problem()
                if failed
                else None,
                0,
            )
        return result

    def run_forever(self) -> None:
        try:
            while not self.stop_event.is_set():
                if self.stop_file and self.stop_file.exists():
                    break
                if not self.run_one():
                    self.stop_event.wait(0.3)
        finally:
            if self.child:
                self.child.close()

    def close(self) -> None:
        self.stop_event.set()
