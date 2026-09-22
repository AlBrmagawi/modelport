# Release verification

Verification date: 2026-09-22. These checks ran locally before initial source
publication. At the time of this audit, no public service was deployed and no
packages or models were uploaded. Current hosted results are available in
[GitHub Actions](https://github.com/AlBrmagawi/modelport/actions/workflows/ci.yml).

The later [pre-push review](pre-push-review.md) records the fresh quality run,
publication checks and fixes made before the initial GitHub push.

## Executed checks

| Gate | Observed result |
| --- | --- |
| Windows Python suite | 106 passed, one Linux-specific skip, 178.23 seconds; malformed-token regression also passes after the fix |
| Precision integration | Actual FP16 weights and static INT8 QDQ; original-source gates passed |
| Public remote importing | MNIST and tiny BERT downloaded through pinned Hub and HTTPS; hashes match; finite CPU predictions |
| Frontend unit tests | Three passed |
| Core Docker demo | Offline nonroot read-only container completed export, optimization, dynamic INT8, three validations, four benchmarks and SDK prediction |
| Python audits | Installed packages and normalized Torch 2.13.0: no known vulnerabilities |
| Python distributions | Wheel/source tarball/source ZIP built; automated content verification checks source correspondence, wheel records, checksums, required files and private-file exclusions; extracted Compose parses |
| Linux Python suite | 106 passed, one Windows-specific skip, 179.12 seconds; nonroot, offline, read-only container |
| Real browser workflow | Passed in 45.5 seconds: upload, plan, conversion, validation, benchmark, downloads, persistence, calibration, static INT8, blocked private URL, accessibility and mobile layout |
| Ruff / mypy / schema / Prettier | Passed; Windows and Linux mypy targets; generated OpenAPI has no drift |
| npm audit and Gitleaks | No known npm vulnerabilities and no detected secrets |
| Installed wheel outside checkout | Bundled UI/notices served; all five actual CPU probes passed in a separate runtime-only environment |
| Windows Docker launcher | Starts the existing Linux engine/service and waits for readiness; tested at port 8765 |
| Docker restart persistence | All eight current model artifacts retained; readiness, bundled UI/notices and five verified CPU probes passed after restart |
| Restricted diagnostic runner | Real 30-sample ONNX CPU benchmark passed; network disabled, root/input files read-only, nonroot |
| Live Docker browser | Authenticated current eight-model library, five probes, desktop/mobile captures in ignored evidence, no JavaScript errors |
| Offline API reference | /docs, /redoc, OpenAPI and local CSS served under the strict CSP; targeted API regression passed |

Two warnings originate in the pinned Starlette httpx test-client and AnyIO aliases.
Warnings are not suppressed. Native tests use actual PyTorch and ONNX Runtime.
Network policy unit tests isolate transport responses; a separate opt-in live test
downloads and executes actual public models.

The extended browser runs found an ambiguous import-source label and a genuine
plan fingerprint issue: JavaScript serializes integral floats as integers. Fingerprints
now normalize equivalent JSON numbers, while changed numerical values still invalidate
the plan. A browser-style API round-trip regression test covers the fix. Concurrent Docker and browser builds exhausted host memory. Heavy checks
now run sequentially. Neither failed attempt is counted as a passing gate.

## Coverage

CPU tests reconstruct SafeTensors bundles, export through the modern exporter,
inspect/execute ONNX, optimize graphs, quantize weights and compare required outputs
with the original source. Dynamic batches and multiple inputs/outputs are tested.
An altered ONNX bias fails validation and blocks normal loading. FP16 preserves
FP32 external types. Static INT8 uses independent calibration and held-out data;
overlap and dataset tampering fail. Fixed-batch mismatches cannot overstate throughput.

Queue tests cover atomic claims, fencing, attempts, cancellation, deadlines, crashes,
publication errors and checkpoints. Retry reuses a real completed export after a
controlled later failure, rehashing all reused bytes. Windows Job Objects and Linux
lifetime controls have platform-specific tests.

Security tests reject unsafe paths/serialization, external-data traversal, excessive
allocations, malformed NPZ/SafeTensors, hostile origins and invalid credentials.
Remote tests reject nonpublic/mixed DNS, private redirects, changed peers, oversized
streams and wrong checksums. API tests use a real worker for auth, dataset inspection,
downloads and ordered SSE replay.

## Actual inference evidence

The precision probe used 512 calibration examples and an independent batch of four.
Static INT8 normalized L2 was about 0.02086 under a 0.05 gate; FP16 about 0.000366
under a 0.005 gate. A smaller calibration set failed; policy was not weakened.
These are synthetic agreement checks, not task accuracy.

| Public example | Inputs | Output |
| --- | --- | --- |
| MNIST ONNX | float32 [1,1,28,28] | [1,10] |
| Tiny random BERT ONNX | Three int64 tensors, each [2,4] | last_hidden_state [2,4,32] |

The first Linux Docker demo recorded these sequential batch-4, one-thread means:
PyTorch 0.0864 ms; FP32 ONNX 0.04767 ms; optimized ONNX 0.04707 ms; dynamic INT8
0.06210 ms. INT8 was smaller but slower on this tiny fixture. Timing depends on host
load and is not a general performance claim.

## Evidence paths

- .tools/pytest.xml: latest Windows suite; pytest-windows-release.xml retains the earlier run.
- .tools/docker-evidence/pytest-linux-prepush.xml: latest Linux suite; pytest-linux.xml retains the earlier run.
- .tools/remote-data/remote-verification.json: live hashes and predictions.
- .tools/docker-evidence/core-demo/ and final-demo/: Linux CPU reports and manifests.
- .tools/docker-evidence/service-verification.json: restart/readiness evidence.
- container-job/results/result.json: actual restricted benchmark.
- .tools/pip-audit.json, torch-audit.json, gitleaks-report.json: audits.
- docs/screenshots/: real captures; web/test-results/: browser diagnostics.
- release/0.1.0/: source ZIP/tarball, wheel, inventory, notices and checksums.

Private data, tokens, downloaded weights, environments and node_modules are excluded
from archives. Generated local evidence stays in ignored directories.

## Boundaries

Windows 10 Pro x64, i5-9400F, 16 GiB RAM, with Docker Desktop Linux/WSL 2. Starting the
stopped engine fixed the initial Docker problem. Python 3.12.14, Torch 2.13.0+cpu,
ONNX 1.23.0 and ORT 1.30.0 are locked. macOS, GPU and remote CI remain unverified.

Trusted PyTorch architectures are vision-mlp and dual-input. BERT is imported ONNX
with explicit tensors, without a generic Transformers loader. Unknown dimensions
need concrete inputs. Custom operators, uploaded code, archives, gated repositories
and additional ML ecosystems are unsupported. No distributed queue, universal native
security sandbox or automatic history pruning is claimed.

The final secret scan excludes generated environments/caches and the two explicit
public documentation/test placeholders. No real credential is allowlisted.
