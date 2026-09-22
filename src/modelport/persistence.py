"""Short SQLite transactions, durable events, fenced attempts, immutable evidence."""

import time
import uuid
from contextlib import contextmanager
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
    insert,
    select,
    update,
)

from modelport.config import Settings
from modelport.domain import TERMINAL, ArtifactBundle, JobRecord, TensorDataset, utcnow
from modelport.errors import ModelPortError
from modelport.security import identity

metadata = MetaData()
datasets = Table(
    "datasets",
    metadata,
    Column("dataset_id", String, primary_key=True),
    Column("data", JSON, nullable=False),
    Column("created_at", String, nullable=False),
)
checkpoints = Table(
    "checkpoints",
    metadata,
    Column("job_id", ForeignKey("jobs.job_id"), primary_key=True),
    Column("step_index", Integer, primary_key=True),
    Column("plan_id", String, nullable=False),
    Column("digest", String, nullable=False),
    Column("data", JSON, nullable=False),
    Column("created_at", String, nullable=False),
)
artifacts = Table(
    "artifacts",
    metadata,
    Column("artifact_id", String, primary_key=True),
    Column("data", JSON, nullable=False),
    Column("created_at", String, nullable=False),
)
jobs = Table(
    "jobs",
    metadata,
    Column("job_id", String, primary_key=True),
    Column("kind", String, nullable=False),
    Column("request", JSON, nullable=False),
    Column("state", String, nullable=False),
    Column("stage", String, nullable=False),
    Column("attempt", Integer, nullable=False, default=0),
    Column("token", String),
    Column("heartbeat", Float),
    Column("result", JSON),
    Column("error", JSON),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("timeout_seconds", Integer, nullable=False),
    Column("idempotency_key", String, unique=True),
    Column("request_hash", String, nullable=False),
    Column("cancel", Boolean, nullable=False, default=False),
)
attempts = Table(
    "attempts",
    metadata,
    Column("job_id", ForeignKey("jobs.job_id"), primary_key=True),
    Column("number", Integer, primary_key=True),
    Column("token", String, nullable=False),
    Column("started_at", String),
    Column("finished_at", String),
    Column("state", String),
    Column("exit_code", Integer),
    Column("pid", Integer),
)
events = Table(
    "events",
    metadata,
    Column("job_id", ForeignKey("jobs.job_id"), primary_key=True),
    Column("sequence", Integer, primary_key=True),
    Column("kind", String, nullable=False),
    Column("payload", JSON, nullable=False),
    Column("created_at", String, nullable=False),
)
reports = Table(
    "reports",
    metadata,
    Column("report_id", String, primary_key=True),
    Column("kind", String, nullable=False),
    Column("artifact_id", ForeignKey("artifacts.artifact_id"), nullable=False),
    Column("data", JSON, nullable=False),
    Column("created_at", String, nullable=False),
)
uploads = Table(
    "uploads",
    metadata,
    Column("upload_id", String, primary_key=True),
    Column("staging_id", String, nullable=False),
    Column("name", String, nullable=False),
    Column("created", Float, nullable=False),
    Column("consumed", Boolean, nullable=False, default=False),
)
settings_table = Table(
    "settings",
    metadata,
    Column("key", String, primary_key=True),
    Column("value", JSON, nullable=False),
)
logs = Table(
    "logs",
    metadata,
    Column("job_id", ForeignKey("jobs.job_id"), primary_key=True),
    Column("text", Text, nullable=False),
)


class Repository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = create_engine(
            f"sqlite:///{(settings.data_dir / 'modelport.db').as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 10},
        )

        @event.listens_for(self.engine, "connect")
        def configure(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")

    def migrate(self) -> None:
        from alembic import command
        from alembic.config import Config

        from modelport import migrations

        config = Config()
        config.set_main_option("script_location", str(migrations.ROOT))
        with self.engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            connection.commit()
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    @contextmanager
    def transaction(self):
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def emit(self, connection, job_id: str, kind: str, payload: dict[str, Any]) -> None:
        sequence = (
            connection.scalar(select(func.max(events.c.sequence)).where(events.c.job_id == job_id))
            or 0
        ) + 1
        connection.execute(
            insert(events).values(
                job_id=job_id, sequence=sequence, kind=kind, payload=payload, created_at=utcnow()
            )
        )

    def submit(
        self, kind: str, request: dict[str, Any], key: str | None = None, timeout: int | None = None
    ) -> JobRecord:
        fingerprint = identity({"kind": kind, "request": request, "timeout": timeout})
        with self.transaction() as connection:
            if key:
                previous = (
                    connection.execute(select(jobs).where(jobs.c.idempotency_key == key))
                    .mappings()
                    .first()
                )
                if previous:
                    if previous["request_hash"] != fingerprint:
                        raise ModelPortError(
                            "IDEMPOTENCY_CONFLICT",
                            "This key was used with a different request",
                            status=409,
                        )
                    return self._record(previous)
            queued = connection.scalar(
                select(func.count())
                .select_from(jobs)
                .where(jobs.c.state.in_(["queued", "running", "cancel_requested"]))
            )
            if queued >= self.settings.max_queued_jobs:
                raise ModelPortError(
                    "RESOURCE_EXHAUSTED", "The bounded job queue is full", status=429
                )
            job_id, now = uuid.uuid4().hex, utcnow()
            connection.execute(
                insert(jobs).values(
                    job_id=job_id,
                    kind=kind,
                    request=request,
                    state="queued",
                    stage="queued",
                    attempt=0,
                    created_at=now,
                    updated_at=now,
                    timeout_seconds=timeout or self.settings.timeout_seconds,
                    idempotency_key=key,
                    request_hash=fingerprint,
                    cancel=False,
                )
            )
            self.emit(connection, job_id, "state", {"state": "queued"})
        return self.job(job_id)

    def _record(self, row) -> JobRecord:
        return JobRecord.model_validate({key: row[key] for key in JobRecord.model_fields})

    def dataset(self, dataset_id: str) -> TensorDataset:
        with self.engine.connect() as connection:
            data = connection.scalar(
                select(datasets.c.data).where(datasets.c.dataset_id == dataset_id)
            )
        if data is None:
            raise ModelPortError("NOT_FOUND", "Dataset does not exist", status=404)
        return TensorDataset.model_validate(data)

    def list_datasets(self) -> list[TensorDataset]:
        with self.engine.connect() as connection:
            return [
                TensorDataset.model_validate(data)
                for data in connection.scalars(
                    select(datasets.c.data).order_by(datasets.c.created_at.desc()).limit(200)
                )
            ]

    def job(self, job_id: str) -> JobRecord:
        with self.engine.connect() as connection:
            row = connection.execute(select(jobs).where(jobs.c.job_id == job_id)).mappings().first()
        if row is None:
            raise ModelPortError("NOT_FOUND", "Job does not exist", status=404)
        return self._record(row)

    def list_jobs(self, limit: int = 50, offset: int = 0) -> list[JobRecord]:
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    select(jobs)
                    .order_by(jobs.c.created_at.desc())
                    .limit(min(limit, 200))
                    .offset(offset)
                )
                .mappings()
                .all()
            )
            return [self._record(row) for row in rows]

    def claim(self) -> dict[str, Any] | None:
        with self.transaction() as connection:
            row = (
                connection.execute(
                    select(jobs)
                    .where(jobs.c.state == "queued")
                    .order_by(jobs.c.created_at, jobs.c.job_id)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            token, number = uuid.uuid4().hex, row["attempt"] + 1
            connection.execute(
                update(jobs)
                .where(jobs.c.job_id == row["job_id"])
                .values(
                    state="running",
                    stage="starting",
                    token=token,
                    heartbeat=time.time(),
                    attempt=number,
                    updated_at=utcnow(),
                )
            )
            connection.execute(
                insert(attempts).values(
                    job_id=row["job_id"],
                    number=number,
                    token=token,
                    started_at=utcnow(),
                    state="running",
                )
            )
            self.emit(connection, row["job_id"], "state", {"state": "running", "attempt": number})
            return {**dict(row), "token": token, "attempt": number}

    def owns(self, connection, job_id: str, token: str) -> bool:
        return (
            connection.scalar(
                select(jobs.c.job_id).where(
                    jobs.c.job_id == job_id,
                    jobs.c.token == token,
                    jobs.c.state == "running",
                    jobs.c.cancel.is_(False),
                )
            )
            is not None
        )

    def heartbeat(self, job_id: str, token: str) -> bool:
        with self.transaction() as connection:
            row = connection.execute(
                select(jobs.c.state, jobs.c.token).where(jobs.c.job_id == job_id)
            ).first()
            if (
                row is None
                or row.token != token
                or row.state not in {"running", "cancel_requested"}
            ):
                return False
            connection.execute(
                update(jobs).where(jobs.c.job_id == job_id).values(heartbeat=time.time())
            )
            return row.state == "running"

    def progress(self, job_id: str, token: str, stage: str, payload: dict[str, Any]) -> None:
        with self.transaction() as connection:
            if self.owns(connection, job_id, token):
                connection.execute(
                    update(jobs)
                    .where(jobs.c.job_id == job_id)
                    .values(stage=stage, updated_at=utcnow())
                )
                self.emit(connection, job_id, "stage", {"stage": stage, **payload})

    def cancel(self, job_id: str) -> JobRecord:
        with self.transaction() as connection:
            row = connection.execute(select(jobs).where(jobs.c.job_id == job_id)).mappings().first()
            if row is None:
                raise ModelPortError("NOT_FOUND", "Job does not exist", status=404)
            if row["state"] not in TERMINAL and row["state"] != "cancel_requested":
                state = "cancelled" if row["state"] == "queued" else "cancel_requested"
                connection.execute(
                    update(jobs)
                    .where(jobs.c.job_id == job_id)
                    .values(state=state, cancel=True, updated_at=utcnow())
                )
                self.emit(connection, job_id, "state", {"state": state})
        return self.job(job_id)

    def retry(self, job_id: str) -> JobRecord:
        with self.transaction() as connection:
            row = connection.execute(select(jobs).where(jobs.c.job_id == job_id)).mappings().first()
            if row is None:
                raise ModelPortError("NOT_FOUND", "Job does not exist", status=404)
            if row["state"] not in {"failed", "cancelled", "timed_out", "interrupted"}:
                raise ModelPortError(
                    "INVALID_TRANSITION",
                    "Only unsuccessful terminal jobs can be retried",
                    status=409,
                )
            connection.execute(
                update(jobs)
                .where(jobs.c.job_id == job_id)
                .values(
                    state="queued",
                    stage="queued",
                    cancel=False,
                    token=None,
                    result=None,
                    error=None,
                    updated_at=utcnow(),
                )
            )
            self.emit(connection, job_id, "state", {"state": "queued", "retry": True})
        return self.job(job_id)

    def finish(
        self,
        connection,
        job_id: str,
        token: str,
        state: str,
        result: Any = None,
        error: Any = None,
        exit_code: int | None = None,
    ) -> bool:
        row = (
            connection.execute(select(jobs).where(jobs.c.job_id == job_id, jobs.c.token == token))
            .mappings()
            .first()
        )
        if row is None or row["state"] not in {"running", "cancel_requested"}:
            return False
        if row["cancel"]:
            state, result, error = "cancelled", None, None
        now = utcnow()
        connection.execute(
            update(jobs)
            .where(jobs.c.job_id == job_id)
            .values(state=state, stage=state, result=result, error=error, updated_at=now)
        )
        connection.execute(
            update(attempts)
            .where(attempts.c.job_id == job_id, attempts.c.token == token)
            .values(state=state, finished_at=now, exit_code=exit_code)
        )
        self.emit(connection, job_id, "state", {"state": state, "result": result, "error": error})
        return True

    def recover(self, stale_after: float = 15) -> int:
        count = 0
        with self.transaction() as connection:
            abandoned = (
                connection.execute(
                    select(jobs).where(
                        jobs.c.state.in_(["running", "cancel_requested"]),
                        jobs.c.heartbeat < time.time() - stale_after,
                    )
                )
                .mappings()
                .all()
            )
            for row in abandoned:
                self.finish(
                    connection,
                    row["job_id"],
                    row["token"],
                    "interrupted",
                    error={
                        "code": "INTERRUPTED",
                        "detail": (
                            "Worker heartbeat expired; retry restarts the native step "
                            "from immutable "
                            "input"
                        ),
                    },
                )
                count += 1
        return count

    def artifact(self, artifact_id: str) -> ArtifactBundle:
        with self.engine.connect() as connection:
            data = connection.scalar(
                select(artifacts.c.data).where(artifacts.c.artifact_id == artifact_id)
            )
        if data is None:
            raise ModelPortError("NOT_FOUND", "Artifact does not exist", status=404)
        return ArtifactBundle.model_validate(data)

    def list_artifacts(self, limit: int = 50, offset: int = 0) -> list[ArtifactBundle]:
        with self.engine.connect() as connection:
            data = connection.scalars(
                select(artifacts.c.data)
                .order_by(artifacts.c.created_at.desc())
                .limit(min(limit, 200))
                .offset(offset)
            ).all()
        return [ArtifactBundle.model_validate(item) for item in data]

    def save_artifact(self, connection, artifact: ArtifactBundle) -> None:
        if (
            connection.scalar(
                select(artifacts.c.artifact_id).where(
                    artifacts.c.artifact_id == artifact.artifact_id
                )
            )
            is None
        ):
            connection.execute(
                insert(artifacts).values(
                    artifact_id=artifact.artifact_id,
                    data=artifact.model_dump(mode="json"),
                    created_at=artifact.created_at,
                )
            )

    def save_report(self, connection, kind: str, data: dict[str, Any], artifact_id: str) -> None:
        connection.execute(
            insert(reports).values(
                report_id=data["report_id"],
                kind=kind,
                artifact_id=artifact_id,
                data=data,
                created_at=data["created_at"],
            )
        )
        if kind == "validation":
            stored = connection.scalar(
                select(artifacts.c.data).where(artifacts.c.artifact_id == artifact_id)
            )
            stored["validation_state"] = data["state"]
            connection.execute(
                update(artifacts).where(artifacts.c.artifact_id == artifact_id).values(data=stored)
            )

    def report(self, report_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            data = connection.scalar(select(reports.c.data).where(reports.c.report_id == report_id))
        if data is None:
            raise ModelPortError("NOT_FOUND", "Report does not exist", status=404)
        return data

    def list_reports(
        self, artifact_id: str | None = None, kind: str | None = None
    ) -> list[dict[str, Any]]:
        query = select(reports.c.data).order_by(reports.c.created_at.desc()).limit(200)
        if artifact_id:
            query = query.where(reports.c.artifact_id == artifact_id)
        if kind:
            query = query.where(reports.c.kind == kind)
        with self.engine.connect() as connection:
            return list(connection.scalars(query))

    def job_events(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
        self.job(job_id)
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(events)
                    .where(events.c.job_id == job_id, events.c.sequence > after)
                    .order_by(events.c.sequence)
                    .limit(100)
                ).mappings()
            ]

    def get_setting(self, key: str) -> Any:
        with self.engine.connect() as connection:
            return connection.scalar(
                select(settings_table.c.value).where(settings_table.c.key == key)
            )

    def set_setting(self, key: str, value: Any) -> None:
        with self.transaction() as connection:
            self.set_setting_in(connection, key, value)

    def set_setting_in(self, connection, key: str, value: Any) -> None:
        from sqlalchemy.dialects.sqlite import insert as upsert

        connection.execute(
            upsert(settings_table)
            .values(key=key, value=value)
            .on_conflict_do_update(index_elements=["key"], set_={"value": value})
        )

    def close(self) -> None:
        self.engine.dispose()
