# Installation and first run

Choose Docker to try the dashboard. Choose the source setup to use the SDK, run
tests or change the application. Both paths run the same CPU services.

## 1. Download the project

With [Git](https://git-scm.com/downloads):

```sh
git clone https://github.com/AlBrmagawi/modelport.git
cd modelport
```

Or [download the source ZIP](https://github.com/AlBrmagawi/modelport/archive/refs/heads/main.zip),
extract it and open a terminal in `modelport-main`. Run the following commands from
the project directory unless a step explicitly changes directories.

## 2A. Run with Docker

### Prerequisites

- [Docker Desktop](https://docs.docker.com/get-started/get-docker/) on Windows,
  configured for Linux containers, or Docker Engine with Compose v2 on Linux.
- Internet for the first build and several GB of disk for libraries and model data.
- The service has a 6 GiB memory limit. Local verification used a 16 GiB host.
  Run heavy builds and tests sequentially on memory-constrained machines.

Check that the engine is running:

```sh
docker version
docker compose version
docker info --format '{{.OSType}}'
```

The last command should print `linux`.

### Start and sign in

```sh
docker compose -f docker/compose.yml up -d --build
docker compose -f docker/compose.yml exec modelport modelport token
```

The first command builds the dashboard, downloads locked CPU libraries and starts
the API and worker. Later builds reuse cached dependency layers. The token command
prints JSON; copy only the value of its `token` field.

On Windows you can instead double-click `start-modelport.cmd`, or run:

```powershell
.\start-modelport.cmd
```

The launcher starts Docker Desktop if needed, waits for readiness, prints the
token and opens the browser. To reuse an already built image:

```powershell
.\start-modelport.cmd -NoBuild
```

Open **http://127.0.0.1:8765**, paste the token and click **Open workspace**.
Allow the environment probes to finish on first run. Check startup with:

```sh
docker compose -f docker/compose.yml ps
docker compose -f docker/compose.yml logs --tail 100
```

### Stop, restart or choose another port

```sh
docker compose -f docker/compose.yml stop
docker compose -f docker/compose.yml up -d
```

The named `modelport-data` volume retains models, reports, jobs and the token.
Ordinary `stop` preserves it; deleting that volume deletes the workspace.

For another port, use the Windows launcher:

```powershell
.\start-modelport.cmd -Port 8767
```

Or in a Linux shell:

```sh
MODELPORT_PORT=8767 docker compose -f docker/compose.yml up -d
```

Open `http://127.0.0.1:8767`. Compose also updates the browser-origin allowlist.

## 2B. Run from source

### Download the prerequisites

| Tool | Download | Purpose |
| --- | --- | --- |
| uv | [Official installation guide](https://docs.astral.sh/uv/getting-started/installation/) | Installs Python and the locked Python libraries |
| Node.js 24 with npm | [Official Node.js download](https://nodejs.org/en/download) | Builds the dashboard |
| Git | [Official Git download](https://git-scm.com/downloads) | Clones and updates the source; optional for a ZIP download |

Python **3.12** is required. `uv sync` can install it automatically. No global
PyTorch, CUDA toolkit, GPU or manually copied runtime library is required.
Open a new terminal after installing the tools and check:

```sh
uv --version
node --version
npm --version
```

Use `npm.cmd --version` in PowerShell if script execution is restricted.

### Windows PowerShell

```powershell
uv sync --locked --extra cpu
cd web
npm.cmd ci
npm.cmd run build
cd ..
.\.venv\Scripts\modelport.exe demo --output .\demo-results
.\.venv\Scripts\modelport.exe token
.\.venv\Scripts\modelport.exe serve
```

Calling `.exe` and `npm.cmd` avoids needing to activate the virtual environment
or change PowerShell's execution policy.

### Linux shell

```sh
uv sync --locked --extra cpu
cd web
npm ci
npm run build
cd ..
uv run --no-sync modelport demo --output ./demo-results
uv run --no-sync modelport token
uv run --no-sync modelport serve
```

Open **http://127.0.0.1:8765** and enter the token. `serve` starts the API, worker
and built dashboard together. Keep the terminal open; use **Ctrl+C** to shut down.

The source setup stores its workspace in `.modelport/`. Docker uses its named
volume. Those are separate workspaces by default.

### How the libraries are installed

`uv sync --locked --extra cpu` creates `.venv` and installs `uv.lock`, including
CPU PyTorch, ONNX, ONNX Runtime and developer tools. `npm ci` installs
`web/package-lock.json`. `npm run build` creates the dashboard under `web/dist/`.

See the [dependency reference](dependencies.md) for the full library overview,
runtime-only setup and audits. Libraries are downloaded from their registries;
environments and `node_modules` are not committed to the repository.

## 3. Test the complete workflow

Click **Run CPU demo** in the dashboard. On a fresh workspace:

1. A deterministic registered PyTorch model appears in the library.
2. FP32 ONNX, optimized ONNX and dynamic INT8 variants are created.
3. Conversion results include numerical validation against the original source.
4. **Benchmarks** shows measured CPU results.
5. Select a model to inspect tensors, validation and provenance, or download its
   artifact. Export benchmark evidence as JSON, CSV or HTML.

The source terminal demo also writes `summary.json`, validation reports, benchmark
reports and artifact ZIPs to `demo-results/`. The model and inputs are synthetic;
they demonstrate the workflow, not real-world classification accuracy. INT8 may
be smaller without being faster.

The token stays in browser memory. Refresh requires entering it again. Recover
the token with the same `modelport token` command for the matching workspace.

## 4. Use your own model

Read [supported formats](capability-matrix.md) and the
[model-bundle contract](model-bundle.md). The registered PyTorch architectures are
`vision-mlp` and `dual-input`; arbitrary Python classes are not reconstructed from
uploaded weights.

Choose **Import model** for supported files. For static INT8, follow the
[calibration and validation dataset guide](datasets.md). Public HTTPS and Hugging
Face imports use the [remote-import guide](remote-imports.md) and its pinned examples.

## 5. Run automated checks

After the source installation:

```sh
uv run --no-sync python scripts/check.py
cd web
npm test
npm run build
npm exec playwright -- install chromium
npm run e2e
```

Use `npm.cmd` in PowerShell. Linux browser tests may require system dependencies:
`npm exec playwright -- install --with-deps chromium`. CPU tests generate local
fixtures and run offline after dependency/browser installation. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for audits and package checks.

## Update an existing checkout

With a clean working tree, run `git pull --ff-only`. For Docker, repeat
`docker compose -f docker/compose.yml up -d --build`. For source, stop the server,
repeat `uv sync --locked --extra cpu`, `npm ci` and `npm run build`, then restart
`modelport serve`. Back up a workspace you rely on before upgrading; see
[backup and restore](deployment.md).

## Troubleshooting

| Symptom | First check |
| --- | --- |
| Docker cannot connect | Start Docker Desktop; check `docker info` and the Linux container engine |
| Port 8765 is occupied | Use another port; avoid starting Docker and source servers on the same port |
| `uv` or Node is not recognized | Reopen the terminal after installing it and check PATH |
| PowerShell blocks `npm.ps1` | Call `npm.cmd` |
| Sign-in is required again | Refresh clears the credential; retrieve it from the matching workspace |
| Unsupported conversion | Read the error and Environment view, then check the capability matrix |
| A build runs out of memory | Stop concurrent heavy jobs and check Docker/host memory allocation |

More: [troubleshooting](troubleshooting.md), [operations](deployment.md) and
[security](../SECURITY.md).
