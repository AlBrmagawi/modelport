import json
import signal
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from modelport.domain import BenchmarkConfig, ConversionPlan, HTTPSImport, HubImport
from modelport.errors import ModelPortError
from modelport.sdk import ModelPort
from modelport.security import read_json

app = typer.Typer(
    help="ModelPort — convert models, verify results, run with confidence.", no_args_is_help=True
)
jobs_app = typer.Typer(help="Durable job history, cancellation and retry.")
artifacts_app = typer.Typer(help="Export immutable artifact bundles and evidence.")
datasets_app = typer.Typer(help="Register explicit calibration and held-out validation tensors.")
app.add_typer(jobs_app, name="jobs")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(datasets_app, name="datasets")


@app.callback()
def configure(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None, typer.Option(help="Overrides MODELPORT_DATA_DIR (default .modelport)")
    ] = None,
):
    ctx.obj = data_dir


def output(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in value
        ]
    typer.echo(json.dumps(value, allow_nan=False, ensure_ascii=True))


def run(ctx: typer.Context, action):
    try:
        with ModelPort.local(ctx.obj) as client:
            output(action(client))
    except ModelPortError as exc:
        output(exc.problem())
        raise typer.Exit(
            4 if exc.code == "VALIDATION_FAILED" else 3 if exc.status == 409 else 2
        ) from exc
    except ValidationError as exc:
        output(
            ModelPortError(
                "INVALID_REQUEST", "Request does not satisfy the documented schema"
            ).problem()
        )
        raise typer.Exit(2) from exc
    except KeyboardInterrupt as exc:
        raise typer.Exit(130) from exc


@app.command()
def doctor(ctx: typer.Context):
    """Run bounded real CPU export, optimization, INT8 and numerical probes."""
    run(ctx, lambda c: c.doctor())


@app.command()
def capabilities(ctx: typer.Context):
    """Show verified support (probes run if evidence is stale)."""
    run(ctx, lambda c: c.capabilities())


@app.command("import")
def import_command(ctx: typer.Context, path: Path, wait: bool = True):
    """Copy and inspect a local file or documented bundle; --no-wait submits."""
    run(ctx, lambda c: c.import_model(path, wait=wait))


@app.command()
def inspect(ctx: typer.Context, artifact_id: str):
    """Read persisted real inspection metadata."""
    run(ctx, lambda c: c.inspect(artifact_id))


@app.command("import-https")
def import_https(ctx: typer.Context, manifest: Path, wait: bool = True):
    """Download public model files from URLs with required SHA-256 checksums."""

    def action(c):
        request = HTTPSImport.model_validate(read_json(manifest, 1024**2))
        return c.import_https([item.model_dump() for item in request.files], wait=wait)

    run(ctx, action)


@app.command("import-hub")
def import_hub(ctx: typer.Context, manifest: Path, wait: bool = True):
    """Download a public Hugging Face bundle pinned to a full commit and checksums."""

    def action(c):
        request = HubImport.model_validate(read_json(manifest, 1024**2))
        return c.import_hub(
            request.repository,
            request.revision,
            [item.model_dump() for item in request.files],
            wait=wait,
        )

    run(ctx, action)


@app.command()
def plan(
    ctx: typer.Context,
    artifact_id: str,
    precision: str = "fp32",
    optimize: bool = False,
    device: str = "cpu",
    skip_validation: bool = False,
    calibration_id: str | None = None,
    validation_id: str | None = None,
    calibration_batch_size: int = 4,
):
    """Print a deterministic plan as JSON; redirect it to a file for convert."""
    run(
        ctx,
        lambda c: c.plan(
            artifact_id,
            precision=precision,
            optimize=optimize,
            device=device,
            skip_validation=skip_validation,
            calibration_id=calibration_id,
            validation_id=validation_id,
            calibration_batch_size=calibration_batch_size,
        ),
    )


@app.command()
def convert(ctx: typer.Context, plan_file: Path, wait: bool = True):
    """Execute a saved JSON plan; stale plans are rejected."""
    run(
        ctx,
        lambda c: c.convert(
            ConversionPlan.model_validate(read_json(plan_file, 4 * 1024**2)), wait=wait
        ),
    )


@app.command()
def validate(
    ctx: typer.Context,
    source_id: str,
    target_id: str,
    inputs: Path | None = None,
    validation_id: str | None = None,
    policy: str = "fp32-default",
    wait: bool = True,
):
    """Compare actual named outputs using NPZ or recorded synthetic inputs."""

    def action(c):
        report = c.validate(
            source_id,
            target_id,
            inputs=inputs,
            validation_id=validation_id,
            policy=policy,
            wait=wait,
        )
        if wait and report.state == "failed":
            output(report)
            raise typer.Exit(4)
        return report

    run(ctx, action)


@app.command()
def benchmark(
    ctx: typer.Context,
    artifact_id: str,
    batch_size: int = 4,
    iterations: int = 30,
    warmup: int = 5,
    threads: int = 1,
    wait: bool = True,
):
    """Measure real CPU inference in a fresh process; timings use milliseconds."""
    run(
        ctx,
        lambda c: c.benchmark(
            artifact_id,
            BenchmarkConfig(
                batch_size=batch_size, iterations=iterations, warmup=warmup, threads=threads
            ),
            wait=wait,
        ),
    )


@app.command()
def predict(ctx: typer.Context, artifact_id: str, inputs: Path):
    """Run inference from a bounded NPZ file; pickle is disabled."""

    def action(c):
        c.repository.artifact(artifact_id)
        job = c.repository.submit(
            "predict", {"source_id": artifact_id, "dataset_id": c.stage_dataset(inputs)}
        )
        return c.wait(job).result

    run(ctx, action)


@jobs_app.command("list")
def jobs_list(ctx: typer.Context, limit: int = 50):
    run(ctx, lambda c: c.repository.list_jobs(limit))


@jobs_app.command("show")
def jobs_show(ctx: typer.Context, job_id: str):
    run(ctx, lambda c: c.repository.job(job_id))


@jobs_app.command("cancel")
def jobs_cancel(ctx: typer.Context, job_id: str):
    run(ctx, lambda c: c.repository.cancel(job_id))


@jobs_app.command("retry")
def jobs_retry(ctx: typer.Context, job_id: str, wait: bool = False):
    run(ctx, lambda c: c.wait(c.repository.retry(job_id)) if wait else c.repository.retry(job_id))


@artifacts_app.command("export")
def artifacts_export(ctx: typer.Context, artifact_id: str, output_path: Path):
    run(ctx, lambda c: {"path": str(c.export_artifact(artifact_id, output_path))})


@datasets_app.command("import")
def dataset_import(ctx: typer.Context, path: Path, purpose: str, wait: bool = True):
    run(ctx, lambda c: c.import_dataset(path, purpose=purpose, wait=wait))


@datasets_app.command("list")
def dataset_list(ctx: typer.Context):
    run(ctx, lambda c: c.repository.list_datasets())


@app.command()
def gc(ctx: typer.Context, apply: bool = False, older_than_hours: int = 24):
    """List unreferenced blobs/staging; --apply deletes eligible owned paths."""
    from modelport.maintenance import garbage_collect

    run(ctx, lambda c: garbage_collect(c, apply=apply, older_than_hours=older_than_hours))


@app.command()
def migrate(ctx: typer.Context):
    """Apply packaged Alembic migrations under the migration lock."""
    run(ctx, lambda c: {"schema": "0003", "data_dir": str(c.settings.data_dir)})


@app.command()
def worker(ctx: typer.Context, stop_file: Path | None = None):
    """Process queued jobs; initialize the data directory with migrate first."""
    from modelport.config import Settings
    from modelport.processes import bind_parent_lifetime

    bind_parent_lifetime()
    with ModelPort(Settings(**({"data_dir": ctx.obj} if ctx.obj else {})), migrate=False) as client:
        client.supervisor.stop_file = stop_file
        signal.signal(signal.SIGTERM, lambda *_: client.supervisor.close())
        try:
            client.supervisor.run_forever()
        except KeyboardInterrupt:
            client.supervisor.close()


@app.command()
def serve(ctx: typer.Context, host: str = "127.0.0.1", port: int = 8765, with_worker: bool = True):
    """Start authenticated API and built dashboard, with a supervised worker."""
    import uvicorn

    from modelport.api import create_app

    uvicorn.run(
        create_app(ctx.obj, start_worker=with_worker), host=host, port=port, log_level="info"
    )


@app.command()
def demo(
    ctx: typer.Context,
    output_dir: Annotated[
        Path, typer.Option("--output", help="Write real manifests and JSON/CSV/HTML reports")
    ] = Path("demo-results"),
    architecture: str = "vision-mlp",
):
    """Run the full offline CPU workflow using deterministic synthetic fixtures."""
    from modelport.demo import run_demo

    run(ctx, lambda c: run_demo(c, output_dir, architecture))


@app.command()
def token(ctx: typer.Context):
    """Show the operator credential from the private local data directory."""
    from modelport.api import operator_token

    run(ctx, lambda c: {"token": operator_token(c.settings)})


if __name__ == "__main__":
    app()
