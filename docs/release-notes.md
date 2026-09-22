# ModelPort 0.1.0 — local CPU release

Release target: downloadable source/package and local Docker application. No public
service or package-registry upload is included. Python 3.12; Windows x64 and Linux
x64 CPU are the tested platforms. macOS and GPU execution are not verified.

## Included

- SDK, CLI, authenticated REST API and bundled React dashboard.
- Registered PyTorch SafeTensors bundles, ONNX inspection and unified CPU inference.
- Real PyTorch export, graph optimization, dynamic/static INT8 and FP16 graph conversion.
- Separate calibration/validation datasets, numerical gates, real benchmark reports.
- Durable jobs, cancellation, interruption recovery, verified step checkpoints and provenance.
- Hardened public HTTPS and commit-pinned Hugging Face downloads; tested MNIST and BERT ONNX.
- Nonroot local Docker app, persistent storage, launcher, tests, CI and operational docs.

## Run the source download

Extract `modelport-0.1.0-source.zip`. On Windows with Docker Desktop installed,
double-click `start-modelport.cmd`. Or, on any supported Docker host, from the extracted directory:

```sh
docker compose -f docker/compose.yml up -d --build
docker compose -f docker/compose.yml exec modelport modelport token
```

Open http://127.0.0.1:8765, paste the token and click **Run CPU demo**. Tokens are
intentionally forgotten on browser refresh. Stop with `docker compose -f docker/compose.yml stop`.

## Install the wheel

The wheel includes the dashboard and needs no Node.js at runtime. Install the CPU
dependencies through the supplied lockfile for a reproducible source installation.
To build the packages from a checkout, run
`uv run --no-sync python scripts/release.py`. A successful GitHub Actions CPU run
also uploads packages under its evidence artifacts. For a standalone wheel, place
the wheel in an installation directory and create a Python 3.12 environment with uv:

```sh
uv venv --python 3.12
uv pip install torch==2.13.0 --index https://download.pytorch.org/whl/cpu
uv pip install "./modelport-0.1.0-py3-none-any.whl[cpu]"
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\modelport.exe demo --output demo-results
.\.venv\Scripts\modelport.exe token
.\.venv\Scripts\modelport.exe serve
```

On Linux:

```sh
.venv/bin/modelport demo --output demo-results
.venv/bin/modelport token
.venv/bin/modelport serve
```

The wheel declares supported dependency ranges; the source archive's `uv.lock`
records exact tested versions. Do not use a bare PyPI `torch` install when a CPU-only
dependency set is required. ModelPort is not assumed to be available on PyPI.

## Verification and limits

See `docs/verification.md` in the source download for executed checks and evidence.
FP16 graph storage does not guarantee FP16 CPU arithmetic. Quantization need not be
faster. Numerical agreement is not task accuracy. Unknown inner dimensions require
explicit tensors. GPU, arbitrary Python architectures, generic Transformers loading,
OpenVINO, TensorFlow/LiteRT, Core ML, TensorRT and GGUF remain outside this release.
The native worker is fault isolation, not a hostile-tenant security sandbox.

Release files include SHA-256 checksums, Apache-2.0 license and third-party notices,
and a dependency inventory. No operator tokens, local databases, downloaded weights,
virtual environments or node_modules are included. `scripts/release.py` rebuilds
the artifacts locally and never publishes them.
