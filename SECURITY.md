# Security boundary

ModelPort is a local, single-operator tool. Do not expose it as a public multitenant
service. SafeTensors avoids pickle execution, but native model parsers and runtimes
can still have vulnerabilities. A child process is fault isolation, not a sandbox.
Use trusted artifacts or an isolated host/container appropriate to your threat model.

Implemented controls:

- Loopback by default; bearer credential on all model/job/artifact/action endpoints.
  Generated tokens live in the private data directory. Browser credentials remain
  in memory and are sent in headers, including fetch-based SSE.
- Host allowlist, exact browser origins, no wildcard CORS, bounded bodies before
  multipart/JSON consumption, per-file and total upload limits, safe error responses.
- No unrestricted API server-local paths, archive extraction,
  pickle loading, uploaded Python, custom operator DLLs, or model-selected plugins.
- POSIX traversal, Windows drives/UNC/ADS, reserved names, case collisions,
  symlinks/junctions/reparse points and external-data ranges are checked.
- SafeTensors/NPZ metadata and claimed tensor allocations are bounded before native
  loading. NPZ uses `allow_pickle=False`. Tensor names, shapes and dtypes are strict.
- Native inspection, export and inference run in supervised processes. Children
  receive allowlisted environment variables, private work directories, bounded
  JSON IPC, separate progress files, bounded/redacted logs, CPU threads and deadlines.
- Windows children start suspended, are assigned to a Job Object with memory and
  kill-on-close limits, then resume. Tree cancellation, timeout and forced owner-death
  cleanup are tested on the verification host. POSIX code uses process groups,
  resource limits and race-checked Linux parent-death signals for workers/runners.
- Data roots get mode 0700 on POSIX. Windows initialization restricts inheritable
  ACLs to the operator SID and SYSTEM. Use an application-owned data directory.
- Immutable artifacts, digest verification, transactional attempts, cancellation
  intent and ownership fencing prevent stale or partial publication.
- HTML escapes metadata; CSV neutralizes formula-prefixed strings. Reports do not
  include operator credentials or full source filesystem paths.

The native child still has the operator's filesystem/network privileges. Windows
Job Objects do not provide Linux-container, VM, or hostile-tenant isolation.
The optional `docker/compose.isolated.yml` profile disables worker networking,
runs nonroot, mounts published inputs read-only, and sets container resource limits;
the actual profile passed a 30-sample CPU benchmark with those restrictions.

Public remote imports validate every initial URL and redirect, reject nonpublic or
mixed DNS answers, connect to a pinned numeric address, check the actual socket peer,
and preserve the original hostname for TLS certificate verification. Only HTTPS/443
is permitted. Ambient proxies, cookies and credentials are not forwarded. Initial
signed URLs are rejected; signed public CDN redirects are allowed and not persisted.
Explicit file SHA-256 hashes, byte budgets, read timeouts and a job deadline apply.
Hub imports require a full commit revision; no repository code or pickle is loaded.
Policy regression tests and actual public downloads are separate verification gates.

Dependency audits are point-in-time evidence, not guarantees. See
[verification.md](docs/verification.md) for exact results and exclusions.
For a vulnerability, use the repository's
[private vulnerability report](https://github.com/AlBrmagawi/modelport/security/advisories/new).
Include reproduction steps, affected versions and the impact. Do not attach real
credentials or sensitive model files to public issues. Version 0.1.x is the
currently maintained beta series; no response-time SLA is promised.

Primary references: [PyTorch security policy](https://github.com/pytorch/pytorch/security/policy),
[ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html),
[ORT graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).
