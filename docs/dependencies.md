# Libraries and dependency installation

ModelPort installs its libraries through version-controlled lockfiles. You do not
need to find runtime DLLs or copy ML libraries into the project manually.

## Install the development environment

From the repository root, with [uv](https://docs.astral.sh/uv/getting-started/installation/)
and [Node.js 24](https://nodejs.org/en/download) installed:

```sh
uv sync --locked --extra cpu
cd web
npm ci
npm run build
```

Use `npm.cmd` in Windows PowerShell. Docker performs these steps automatically;
follow the [installation guide](getting-started.md) for that path.

## Python libraries

| Library | Responsibility |
| --- | --- |
| PyTorch CPU | Registered architectures, source inference and ONNX export |
| ONNX / ONNX Script | Graph representation, inspection and modern export support |
| ONNX Runtime CPU | Graph execution, optimization and quantization |
| SafeTensors | Registered weights without pickle deserialization |
| NumPy | Tensor inputs, NPZ datasets and numerical comparisons |
| FastAPI / Uvicorn / Pydantic | Authenticated API, server and typed contracts |
| SQLAlchemy / Alembic | Persistent records and database migrations |
| Typer / Rich | Command-line interface |
| HTTPX / HTTPCore | HTTP clients and constrained public download transport |
| psutil | Process supervision and resource measurements |
| Pytest / Ruff / mypy / pip-audit | Tests, formatting/linting, types and audits |

The tested core versions are Python 3.12.14, Torch 2.13.0+cpu, ONNX 1.23.0,
ONNX Runtime 1.30.0, ONNX Script 0.7.2 and NumPy 2.5.3. The complete exact set is in
[`uv.lock`](../uv.lock); direct-dependency ranges are in
[`pyproject.toml`](../pyproject.toml).

The explicit PyTorch index is `https://download.pytorch.org/whl/cpu`, configured
in `pyproject.toml`. The documented installation uses CPU dependencies and does
not require CUDA.

## Frontend libraries

| Library | Responsibility |
| --- | --- |
| React / React DOM | Dashboard and interactive workflows |
| Lucide React | Interface icons |
| TypeScript / Vite | Type checking and production build |
| openapi-typescript | Generated API types |
| Vitest | Frontend unit tests |
| Playwright / axe-core | Real browser workflow and accessibility checks |
| Prettier | Frontend formatting |

[`web/package-lock.json`](../web/package-lock.json) records exact versions and
integrity hashes. `npm ci` installs that set. The npm package is private; its built
dashboard is distributed inside ModelPort.

## Runtime-only setup

Omit developer tools when running the application:

```sh
uv sync --locked --extra cpu --no-dev
```

Build the dashboard, then run `uv run --no-sync modelport serve`. Restore developer
tools with `uv sync --locked --extra cpu` when you want to run tests.

A built wheel includes the dashboard and needs no Node at runtime. Follow the
[wheel instructions](release-notes.md#install-the-wheel) for the explicit CPU
installation. The source lockfile is more reproducible than installing a standalone
wheel's dependency ranges.

## Check and audit

```sh
uv pip check
uv run --no-sync python scripts/audit.py
cd web
npm audit --audit-level=low
```

The Python script audits installed packages and separately checks Torch's upstream
version because advisory services may not recognize the `+cpu` suffix. ModelPort
itself is unpublished on PyPI. Audit results reflect known advisories at scan time.

## Updating dependencies

Update the relevant manifest and lockfile together, then run the quality sequence
in [CONTRIBUTING.md](../CONTRIBUTING.md). ML runtime updates must pass actual
conversion, inference, numerical validation, quantization, browser and Docker checks.

Release builds include an inventory and notices. Model weights are separate assets
with their own terms and are not shipped as library dependencies. See
[third-party notices](../THIRD_PARTY_NOTICES.md).
