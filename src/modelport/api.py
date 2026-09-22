"""Authenticated local REST interface. Native model work is queued, never parsed here."""

import asyncio
import hmac
import os
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field
from sqlalchemy import insert, select
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from modelport.config import Settings
from modelport.domain import (
    TERMINAL,
    ArtifactBundle,
    BenchmarkConfig,
    BenchmarkReport,
    CapabilitySnapshot,
    Contract,
    ConversionPlan,
    ConversionRequest,
    HTTPSImport,
    HubImport,
    JobRecord,
    ModelDescriptor,
    TensorDataset,
    ValidationReport,
)
from modelport.errors import ModelPortError
from modelport.persistence import attempts, logs, uploads
from modelport.planning import build_plan
from modelport.processes import ChildProcess
from modelport.reporting import benchmark_csv, benchmark_html
from modelport.sdk import ModelPort
from modelport.security import canonical, safe_relative


class Problem(Contract):
    type: str
    title: str
    status: int
    code: str
    detail: str
    request_id: str
    job_id: str | None = None
    retryable: bool
    details: dict[str, Any]


class ImportRequest(Contract):
    upload_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class FixtureRequest(Contract):
    architecture: Literal["vision-mlp", "dual-input"] = "vision-mlp"


class ValidationRequest(Contract):
    source_id: str
    target_id: str
    policy: Literal["fp32-default", "int8-synthetic-v1", "int8-calibrated-v1", "fp16-default"] = (
        "fp32-default"
    )
    validation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    mapping: dict[str, str] | None = None
    batch_sizes: list[int] = Field(default_factory=lambda: [2, 4], min_length=1, max_length=8)
    seed: int = Field(default=2027, ge=0, le=2**32 - 1)


class BenchmarkRequest(Contract):
    artifact_id: str
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


def operator_token(settings: Settings) -> str:
    supplied = os.getenv("MODELPORT_TOKEN")
    if supplied:
        if len(supplied) < 24:
            raise ModelPortError(
                "INVALID_CONFIG", "MODELPORT_TOKEN must contain at least 24 characters"
            )
        return supplied
    path = settings.data_dir / "operator-token"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(secrets.token_urlsafe(32))
    except FileExistsError:
        pass
    return path.read_text(encoding="ascii").strip()


class RequestBoundary:
    def __init__(self, app, max_bytes: int, origins: list[str]):
        self.app, self.max_bytes, self.origins = app, max_bytes, origins

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        error = None
        origin = headers.get(b"origin", b"").decode("latin1")
        if origin and origin not in self.origins:
            error = ModelPortError("ORIGIN_REJECTED", "Browser origin is not allowed", status=403)
        limit = self.max_bytes if scope["path"].endswith(("/uploads", "/datasets")) else 4 * 1024**2
        try:
            length = int(headers.get(b"content-length", b"0"))
            if length < 0 or length > limit:
                error = ModelPortError(
                    "REQUEST_TOO_LARGE", "Request body exceeds the configured limit", status=413
                )
        except ValueError:
            error = ModelPortError("INVALID_REQUEST", "Invalid Content-Length", status=400)
        if error:
            return await JSONResponse(error.problem(request_id), status_code=error.status)(
                scope, receive, send
            )
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > limit:
                raise ModelPortError(
                    "REQUEST_TOO_LARGE", "Streaming request exceeds body limit", status=413
                )
            return message

        async def safe_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"cache-control", b"no-store"),
                        (
                            b"content-security-policy",
                            b"default-src 'self'; script-src 'self'; style-src 'self'; "
                            b"connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
                        ),
                        (b"x-request-id", request_id.encode()),
                    ]
                )
            await send(message)

        return await self.app(scope, bounded_receive, safe_send)


def create_app(data_dir: Path | str | None = None, *, start_worker: bool = True) -> FastAPI:
    client = ModelPort.local(data_dir)
    token = operator_token(client.settings)
    origins = os.getenv(
        "MODELPORT_ALLOWED_ORIGINS",
        "http://127.0.0.1:8765,http://localhost:8765,http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    worker: ChildProcess | None = None
    worker_dir: Path | None = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal worker, worker_dir
        if start_worker:
            worker_dir = client.settings.data_dir / "work" / ("service-" + uuid.uuid4().hex)
            worker_dir.mkdir(mode=0o700)
            worker = ChildProcess(
                [
                    "-m",
                    "modelport.cli",
                    "--data-dir",
                    str(client.settings.data_dir),
                    "worker",
                    "--stop-file",
                    str(worker_dir / "stop"),
                ],
                worker_dir,
                6 * 1024**3,
            )
            try:
                client.capabilities(probe=False)
            except ModelPortError:
                client.repository.submit("doctor", {})
        yield
        if worker:
            assert worker_dir is not None
            (worker_dir / "stop").touch()
            for _ in range(50):
                if worker.process.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            worker.close()
        if (
            worker_dir
            and worker_dir.resolve().parent == (client.settings.data_dir / "work").resolve()
        ):
            shutil.rmtree(worker_dir, ignore_errors=True)
        client.close()

    app = FastAPI(
        title="ModelPort",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        description=(
            "Local single-operator model workflows. Bearer authentication required. "
            "Expensive operations return durable jobs; follow /jobs/{id}/events with "
            "Authorization and Last-Event-ID headers."
        ),
        lifespan=lifespan,
        responses={
            code: {"model": Problem} for code in (400, 401, 403, 404, 409, 413, 422, 429, 503)
        },
    )
    app.state.client = client
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Last-Event-ID"],
        expose_headers=["Location", "X-Request-ID"],
    )
    app.add_middleware(
        RequestBoundary, max_bytes=client.settings.max_bundle_bytes + 1024**2, origins=origins
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.getenv("MODELPORT_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(
            ","
        ),
    )
    bearer = HTTPBearer(auto_error=False)

    async def authorized(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        if credentials is None or not hmac.compare_digest(
            credentials.credentials.encode("utf-8"), token.encode("utf-8")
        ):
            raise ModelPortError(
                "UNAUTHORIZED", "A valid operator bearer credential is required", status=401
            )

    auth = [Depends(authorized)]

    @app.exception_handler(ModelPortError)
    async def modelport_error(request: Request, exc: ModelPortError):
        return JSONResponse(
            exc.problem(getattr(request.state, "request_id", "")), status_code=exc.status
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            ModelPortError(
                "INVALID_REQUEST",
                (
                    "Request does not match the documented schema; check field types, bounds "
                    "and supported options"
                ),
            ).problem(getattr(request.state, "request_id", "")),
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        error = ModelPortError(
            "INVALID_HTTP_REQUEST",
            "Request routing or multipart parsing failed; check the endpoint and upload limits",
            status=exc.status_code,
        )
        return JSONResponse(
            error.problem(getattr(request.state, "request_id", "")), status_code=exc.status_code
        )

    def queued(job: JobRecord):
        return JSONResponse(
            job.model_dump(mode="json"),
            status_code=202,
            headers={"Location": f"/api/v1/jobs/{job.job_id}"},
        )

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        status = client.repository.get_setting("worker") or {}
        okay = time.time() - status.get("heartbeat", 0) < 5
        return JSONResponse(
            {"status": "ready" if okay else "worker_unavailable"}, status_code=200 if okay else 503
        )

    @app.get("/api/v1/capabilities", dependencies=auth, response_model=CapabilitySnapshot)
    def capabilities():
        return client.capabilities(probe=False)

    @app.post("/api/v1/doctor", dependencies=auth, status_code=202, response_model=JobRecord)
    def doctor(idempotency_key: Annotated[str | None, Header(max_length=128)] = None):
        return queued(client.repository.submit("doctor", {}, idempotency_key))

    @app.post("/api/v1/uploads", dependencies=auth, status_code=201)
    async def upload(request: Request):
        identifier = uuid.uuid4().hex
        directory = client.settings.data_dir / "staging" / identifier
        directory.mkdir(mode=0o700)
        count, total, seen = 0, 0, set()
        try:
            async with request.form(
                max_files=client.settings.max_files, max_fields=0, max_part_size=65536
            ) as form:
                for key, item in form.multi_items():
                    if key != "files" or not isinstance(item, UploadFile):
                        raise ModelPortError("INVALID_UPLOAD", "Use multipart files fields")
                    filename = safe_relative(item.filename or "")
                    if "/" in filename or filename.casefold() in seen:
                        raise ModelPortError(
                            "UNSAFE_PATH",
                            "Upload unique root bundle filenames; nested upload paths are disabled",
                        )
                    if Path(filename).suffix.lower() not in {
                        ".onnx",
                        ".json",
                        ".safetensors",
                        ".data",
                    }:
                        raise ModelPortError(
                            "UNSUPPORTED_FORMAT",
                            (
                                "Upload ONNX, SafeTensors or documented bundle files; "
                                "code, pickle and "
                                "archives are rejected"
                            ),
                        )
                    seen.add(filename.casefold())
                    count += 1
                    size = 0
                    with (directory / filename).open("xb") as stream:
                        while chunk := await item.read(1024**2):
                            if size == 0 and chunk.startswith(
                                b"version https://git-lfs.github.com/spec/"
                            ):
                                raise ModelPortError(
                                    "LFS_POINTER",
                                    "Upload actual model bytes, not Git LFS pointer files",
                                )
                            size += len(chunk)
                            total += len(chunk)
                            if (
                                size > client.settings.max_file_bytes
                                or total > client.settings.max_bundle_bytes
                            ):
                                raise ModelPortError(
                                    "RESOURCE_EXHAUSTED",
                                    "Upload exceeds the file or bundle byte limit",
                                    status=413,
                                )
                            stream.write(chunk)
            if count == 0:
                raise ModelPortError("INVALID_UPLOAD", "Select at least one model file")
            with client.repository.transaction() as connection:
                connection.execute(
                    insert(uploads).values(
                        upload_id=identifier,
                        staging_id=identifier,
                        name=next(iter(sorted(seen))),
                        created=time.time(),
                        consumed=False,
                    )
                )
            return {
                "upload_id": identifier,
                "files": count,
                "size_bytes": total,
                "expires_in_seconds": 86400,
            }
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    @app.post("/api/v1/imports/https", dependencies=auth, status_code=202, response_model=JobRecord)
    def import_https(body: HTTPSImport):
        return queued(client.import_https([item.model_dump() for item in body.files], wait=False))

    @app.post(
        "/api/v1/imports/huggingface", dependencies=auth, status_code=202, response_model=JobRecord
    )
    def import_hub(body: HubImport):
        return queued(
            client.import_hub(
                body.repository,
                body.revision,
                [item.model_dump() for item in body.files],
                wait=False,
            )
        )

    @app.get("/api/v1/datasets", dependencies=auth, response_model=list[TensorDataset])
    def dataset_list():
        return client.repository.list_datasets()

    @app.post("/api/v1/datasets", dependencies=auth, status_code=202, response_model=JobRecord)
    async def dataset_upload(request: Request, purpose: Literal["calibration", "validation"]):
        identifier = uuid.uuid4().hex
        directory = client.settings.data_dir / "staging" / identifier
        directory.mkdir(mode=0o700)
        try:
            async with request.form(max_files=1, max_fields=0, max_part_size=65536) as form:
                items = list(form.multi_items())
                if (
                    len(items) != 1
                    or items[0][0] != "file"
                    or not isinstance(items[0][1], UploadFile)
                ):
                    raise ModelPortError("INVALID_UPLOAD", "Upload one NPZ using the file field")
                item = items[0][1]
                filename = safe_relative(item.filename or "")
                if "/" in filename or Path(filename).suffix.lower() != ".npz":
                    raise ModelPortError(
                        "UNSUPPORTED_FORMAT", "Dataset uploads require one root NPZ filename"
                    )
                size = 0
                with (directory / "inputs.npz").open("xb") as stream:
                    while chunk := await item.read(1024**2):
                        size += len(chunk)
                        if size > client.settings.max_file_bytes:
                            raise ModelPortError(
                                "RESOURCE_EXHAUSTED", "Dataset exceeds byte limit", status=413
                            )
                        stream.write(chunk)
            return queued(
                client.repository.submit("dataset", {"staging_id": identifier, "purpose": purpose})
            )
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    @app.post("/api/v1/imports", dependencies=auth, status_code=202, response_model=JobRecord)
    def import_upload(
        body: ImportRequest, idempotency_key: Annotated[str | None, Header(max_length=128)] = None
    ):
        with client.repository.engine.connect() as connection:
            row = (
                connection.execute(select(uploads).where(uploads.c.upload_id == body.upload_id))
                .mappings()
                .first()
            )
        if row is None or time.time() - row["created"] > 86400:
            raise ModelPortError("NOT_FOUND", "Upload is missing or expired", status=404)
        return queued(
            client.repository.submit(
                "import", {"staging_id": row["staging_id"], "name": row["name"]}, idempotency_key
            )
        )

    @app.post("/api/v1/fixtures", dependencies=auth, status_code=202, response_model=JobRecord)
    def fixture(
        body: FixtureRequest, idempotency_key: Annotated[str | None, Header(max_length=128)] = None
    ):
        return queued(
            client.repository.submit("fixture", body.model_dump(mode="json"), idempotency_key)
        )

    @app.get("/api/v1/models", dependencies=auth, response_model=list[ArtifactBundle])
    def models(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        return client.repository.list_artifacts(limit, offset)

    @app.get("/api/v1/models/{artifact_id}", dependencies=auth, response_model=ArtifactBundle)
    @app.get("/api/v1/artifacts/{artifact_id}", dependencies=auth, response_model=ArtifactBundle)
    def artifact(artifact_id: str):
        return client.repository.artifact(artifact_id)

    @app.get(
        "/api/v1/models/{artifact_id}/inspection", dependencies=auth, response_model=ModelDescriptor
    )
    def inspect(artifact_id: str):
        return client.inspect(artifact_id)

    @app.post("/api/v1/plans", dependencies=auth, response_model=ConversionPlan)
    def plan(body: ConversionRequest):
        return build_plan(
            client.repository.artifact(body.source_id),
            body,
            client.capabilities(probe=False),
            client.plan_datasets(body),
        )

    @app.post("/api/v1/conversions", dependencies=auth, status_code=202, response_model=JobRecord)
    def convert(
        body: ConversionPlan, idempotency_key: Annotated[str | None, Header(max_length=128)] = None
    ):
        return queued(client.convert(body, wait=False, idempotency_key=idempotency_key))

    @app.get("/api/v1/jobs", dependencies=auth, response_model=list[JobRecord])
    def job_list(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        return client.repository.list_jobs(limit, offset)

    @app.get("/api/v1/jobs/{job_id}", dependencies=auth, response_model=JobRecord)
    def job_show(job_id: str):
        return client.repository.job(job_id)

    @app.get("/api/v1/jobs/{job_id}/attempts", dependencies=auth)
    def job_attempts(job_id: str):
        client.repository.job(job_id)
        with client.repository.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(
                        attempts.c.number,
                        attempts.c.started_at,
                        attempts.c.finished_at,
                        attempts.c.state,
                        attempts.c.exit_code,
                    )
                    .where(attempts.c.job_id == job_id)
                    .order_by(attempts.c.number)
                ).mappings()
            ]

    @app.post("/api/v1/jobs/{job_id}/cancel", dependencies=auth, response_model=JobRecord)
    def job_cancel(job_id: str):
        return client.repository.cancel(job_id)

    @app.post(
        "/api/v1/jobs/{job_id}/retry", dependencies=auth, status_code=202, response_model=JobRecord
    )
    def job_retry(job_id: str):
        return queued(client.repository.retry(job_id))

    @app.get("/api/v1/jobs/{job_id}/logs", dependencies=auth)
    def job_logs(job_id: str):
        client.repository.job(job_id)
        with client.repository.engine.connect() as connection:
            return {
                "text": connection.scalar(select(logs.c.text).where(logs.c.job_id == job_id)) or "",
                "limit_bytes": client.settings.max_log_bytes,
            }

    @app.get("/api/v1/jobs/{job_id}/events", dependencies=auth, response_class=StreamingResponse)
    async def job_events(
        job_id: str, request: Request, last_event_id: Annotated[int, Header(ge=0)] = 0
    ):
        client.repository.job(job_id)

        # Important events are retained for the lifetime of their job; cursors never silently reset.
        async def stream():
            cursor = last_event_id
            while not await request.is_disconnected():
                events = client.repository.job_events(job_id, cursor)
                for item in events:
                    cursor = item["sequence"]
                    yield f"id: {cursor}\nevent: {item['kind']}\ndata: {canonical(item)}\n\n"
                if client.repository.job(job_id).state in TERMINAL and len(events) < 100:
                    return
                if not events:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.4)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @app.post("/api/v1/validations", dependencies=auth, status_code=202, response_model=JobRecord)
    def validation(
        body: ValidationRequest,
        idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
    ):
        return queued(
            client.validate(**body.model_dump(), wait=False, idempotency_key=idempotency_key)
        )

    @app.get("/api/v1/validations/{report_id}", dependencies=auth, response_model=ValidationReport)
    def validation_report(report_id: str):
        report = client.repository.report(report_id)
        if "outputs" not in report:
            raise ModelPortError("NOT_FOUND", "Validation report does not exist", status=404)
        return report

    @app.post("/api/v1/benchmarks", dependencies=auth, status_code=202, response_model=JobRecord)
    def benchmark(
        body: BenchmarkRequest,
        idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
    ):
        return queued(
            client.benchmark(
                body.artifact_id, body.config, wait=False, idempotency_key=idempotency_key
            )
        )

    @app.get("/api/v1/benchmarks/{report_id}", dependencies=auth, response_model=BenchmarkReport)
    def benchmark_report(report_id: str):
        report = client.repository.report(report_id)
        if "measurements" not in report:
            raise ModelPortError("NOT_FOUND", "Benchmark report does not exist", status=404)
        return report

    @app.get("/api/v1/reports", dependencies=auth)
    def report_list(
        artifact_id: str | None = None, kind: Literal["validation", "benchmark"] | None = None
    ):
        return client.repository.list_reports(artifact_id, kind)

    @app.get("/api/v1/benchmarks/{report_id}/export", dependencies=auth)
    def report_export(report_id: str, format: Literal["json", "csv", "html"] = "json"):
        report = client.repository.report(report_id)
        if "measurements" not in report:
            raise ModelPortError("NOT_FOUND", "Benchmark report does not exist", status=404)
        body = (
            canonical(report)
            if format == "json"
            else benchmark_csv([report])
            if format == "csv"
            else benchmark_html([report])
        )
        return Response(
            body,
            media_type={"json": "application/json", "csv": "text/csv", "html": "text/html"}[format],
            headers={"Content-Disposition": f'attachment; filename="benchmark.{format}"'},
        )

    @app.get("/api/v1/artifacts/{artifact_id}/download", dependencies=auth)
    def download(artifact_id: str):
        client.repository.artifact(artifact_id)
        path = client.settings.data_dir / "uploads" / (uuid.uuid4().hex + ".zip")
        client.export_artifact(artifact_id, path)
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"modelport-{artifact_id[:12]}.zip",
            background=BackgroundTask(path.unlink, missing_ok=True),
        )

    @app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/redoc", response_class=HTMLResponse, include_in_schema=False)
    def api_reference():
        from modelport.reference import render_reference

        return render_reference(app.openapi())

    @app.get("/api-reference.css", include_in_schema=False)
    def api_reference_css():
        from modelport.reference import CSS

        return Response(CSS, media_type="text/css")

    bundled = Path(__file__).resolve().parent / "web"
    default_web = (
        bundled if bundled.is_dir() else Path(__file__).resolve().parents[2] / "web" / "dist"
    )
    frontend = Path(os.getenv("MODELPORT_WEB_DIR", str(default_web)))
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="dashboard")
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def dashboard_missing():
            return (
                "<h1>ModelPort</h1><p>Build the dashboard with npm ci &amp;&amp; npm run "
                "build in web/.</p>"
            )

    return app
