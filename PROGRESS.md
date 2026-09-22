# ModelPort release progress

Publication target: downloadable source/package and local Docker application.
Work is isolated in modelport; unrelated projects and containers are preserved.
This checklist records local release acceptance. Source publication is handled
through GitHub; no public ModelPort service is deployed.

## Implemented

- [x] Locked Python 3.12 CPU toolchain and architecture/security documentation.
- [x] Safe local/browser imports, ONNX inspection, SafeTensors and two trusted architectures.
- [x] Real PyTorch export, ONNX optimization and dynamic INT8.
- [x] Original-source numerical validation, unified CPU inference and real benchmarks.
- [x] Durable queue, cancellation, deadlines, attempt fencing and interruption recovery.
- [x] Digest-verified checkpoints, immutable provenance and bounded worker IPC.
- [x] Shared SDK, CLI, authenticated REST API, live events and React dashboard.
- [x] Explicit calibration/validation NPZ registry and actual static INT8 QDQ.
- [x] FP16 graph conversion, preserved FP32 external tensors and CPU execution.
- [x] Public HTTPS with DNS/socket pinning, redirect checks and file hashes.
- [x] Full-revision Hub imports; actual MNIST and tiny BERT ONNX predictions.
- [x] Dashboard dataset controls and remote import forms.
- [x] Fixed-batch benchmark accounting and reserved-name NPZ round trips.
- [x] Start Docker Desktop's Linux engine; build and run the core image/demo.
- [x] Nonroot read-only Compose service, persistent volume and Windows launcher.
- [x] Dashboard in the wheel; source ZIP/tarball, inventory, checksums and notices.

## Release verification

- [x] Windows: 106 Python tests passed; one Linux-specific test skipped.
- [x] Public HTTPS and Hub imports produced identical immutable artifacts.
- [x] Core Docker demo: three validated conversions, four benchmarks, SDK [4,10].
- [x] Python dependency and normalized Torch audits: no known vulnerabilities.
- [x] Real browser workflow: calibration/static INT8, remote policy, accessibility and mobile checks.
- [x] Linux suite: 106 passed, one Windows-specific skip, read-only offline nonroot container.
- [x] Restricted diagnostic runner: actual offline, read-only, nonroot benchmark.
- [x] Installed-wheel smoke, launcher and Docker restart persistence.
- [x] Archive review: checksums match, wheel matches source, no private files, extracted Compose parses.
- [x] Final source ZIP/tarball, bundled wheel, notices, inventory and SHA-256 checksums generated.

## Scope

All mandatory first-release workflows are implemented. Added remote, calibration
and FP16 features use real libraries and outputs. BERT executes an existing ONNX
graph; generic Transformers/SafeTensors reconstruction is not claimed. OpenVINO,
TensorFlow/LiteRT, Core ML, TensorRT, GGUF and GPU remain optional future adapters.
No simulated timings, public hostile-tenant sandbox, distributed queue or automatic
history deletion is claimed.

See [verification](docs/verification.md), [capabilities](docs/capability-matrix.md),
[release notes](docs/release-notes.md) and [security](SECURITY.md).
