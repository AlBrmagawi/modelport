# Environment assessment

Initial assessment: 2026-09-21. Workspace: `D:/AI-PROJECT`, initially empty and not
a Git repository. Shell: Windows PowerShell; script execution is restricted, so
frontend commands use `npm.cmd`.

| Property | Observed |
| --- | --- |
| OS | Windows 10 Pro, x64 |
| CPU | Intel Core i5-9400F, 6 physical / 6 logical cores |
| RAM | 16,705,648 KiB total; approximately 8 GiB initially free |
| Free workspace disk | 50,391,416,832 bytes initially |
| Node / npm | 24.18.0 / 11.16.0 |
| Python | Windows Store alias only; no usable Python or py/uv on PATH |
| GPU | NVIDIA RTX 2060, 6144 MiB, driver 610.74 (listing only) |
| Docker | CLI 29.6.2; Linux daemon unavailable, named pipe absent |

Toolchain decision: project virtual environment, Python 3.12 managed by uv, PyTorch
CPU wheel index, ONNX and ONNX Runtime CPU. GPU is deliberately not an executable
target. No claim of GPU inference is made. No optional ML ecosystems installed.

Windows has no POSIX rlimits. Workers will use process-tree termination and Windows
Job Objects where supported; these controls do not constitute a security sandbox.
Docker runtime/build checks were initially blocked by the stopped Desktop engine;
see the follow-up verification below.

Compatibility references consulted: [PyTorch modern exporter](https://docs.pytorch.org/docs/stable/onnx.html),
[ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html),
[uv Python installation](https://docs.astral.sh/uv/guides/install-python/).
Installed versions and real execution evidence will be appended after verification.

## Verified CPU toolchain

| Component | Installed and executed version |
| --- | --- |
| uv | 0.12.17 |
| CPython | 3.12.14, x64, project `.venv` |
| PyTorch | 2.13.0+cpu, explicit CPU wheel index |
| ONNX | 1.23.0 |
| ONNX Runtime | 1.30.0 |
| ONNX Script | 0.7.2 |
| SafeTensors | 0.8.0 |
| NumPy | 2.5.3 |
| FastAPI / SQLAlchemy / Alembic | 0.141.1 / 2.0.54 / 1.20.0 |
| React / Vite / TypeScript | Exact frontend versions in `web/package-lock.json` |

CPUExecutionProvider executed the actual exported, optimized and INT8 graphs.
AzureExecutionProvider is also advertised by the package, but was not selected or
verified. PyTorch CUDA availability is false with this CPU wheel. Device requests
never fall back silently. Six physical/logical CPU cores and 17,106,583,552 bytes
of physical RAM were reported by the runtime environment snapshot.

The original PyTorch 2.10.0 CPU probe worked, but an upstream-version audit found
PYSEC-2026-139 and PYSEC-2025-194. The CPU build suffix was skipped by the ordinary
installed-package audit. The final lock upgrades to 2.13.0+cpu, and `scripts/audit.py`
audits both installed packages and the normalized upstream Torch version. Both
final scans found no known advisories. The local ModelPort package is unpublished
and therefore has no PyPI vulnerability record.

Installed PyTorch 2.13 removed the `fallback` export parameter; its actual signature
was inspected and the call updated. ModelPort still uses dynamo explicitly and
implements no legacy exporter fallback. Exporter optimization folds constant
weight transposes. ORT quantization uses ONNX shape inference rather than ORT's
symbolic Constant handler, and reduced-range QInt8 avoids observed U8S8 saturation
on this CPU. These effective options are recorded in every plan/manifest.

A separately created `.tools/clean-venv` installed the same locked runtime packages
without developer tools and completed the full demo. It subsequently installed the
built wheel non-editably and passed migrations plus all three real native probes.
Windows Job Object suspension/assignment, memory limits,
tree termination and forced supervisor-death cleanup are covered by tests. POSIX
limits are implemented and Linux-target type checking is performed, but native
Linux execution became available after starting Docker Desktop; macOS remains untested.

## Docker follow-up, 2026-09-22

Docker Desktop 4.83 was installed but not running. Starting it restored the Linux
engine (Docker 29.6.2, WSL 2 kernel 6.18.33.2). The actual Linux image built and its
offline CPU demo passed all three conversion validations and SDK prediction `[4,10]`.
The Compose app became healthy. Unrelated existing Docker projects were preserved.
Container storage already points to D: through the user's Docker data junction.

This 16 GiB machine runs other applications and Docker services. Concurrent Windows
browser checks and Docker builds exhausted host commit memory; C: also has little
space for pagefile growth. Remaining release checks run sequentially. The observed
resource failure is recorded rather than treated as a passing browser test.
Current test counts and final container results are in [verification](verification.md).
