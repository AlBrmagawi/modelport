# Security implementation notes

The authoritative threat boundary and implemented controls are in
[SECURITY.md](../SECURITY.md). Security regression tests cover:

- Platform-independent dangerous path syntax and rejected executable formats.
- Git LFS pointers, upload limits, duplicate metadata, oversized SafeTensors headers,
  truncated tensors, object NPZ arrays and forged enormous NPY allocation headers.
- ONNX external-data traversal, missing dependencies and out-of-range lengths.
- Credentials absent from child environments, authentication, hostile browser origins,
  forbidden API local paths, strict request schemas, and artifact authorization.
- HTML escaping, CSV formula neutralization, stale attempt fencing, process-tree
  cancellation/timeouts and Windows/Linux owner-death handling.
- Remote private/reserved IP ranges, mixed public/private DNS answers, socket peer
  checks, private redirects, checksum mismatches and oversized download streams.
- Calibration/validation sample overlap and dataset integrity at execution time.

The operator can intentionally access local paths through the SDK/CLI. This is a
different trust boundary from browser input. Source model files are copied before
work, but a malicious actor with the operator's own OS account remains out of scope.

Resource defaults are in `config.py`: 64 files, 128 MiB/file, 256 MiB/bundle,
1 MiB metadata, 128 MiB normalized tensor bytes, 256 KiB log output, 4 MiB JSON IPC,
180 seconds/job, 4 GiB native job memory, 64 pending/running jobs, one CPU thread.
The supervisor's Windows service job has a 6 GiB aggregate memory budget, including
its current native child. Limits reduce accidental resource exhaustion; they are
not proof against all parser or kernel vulnerabilities.
