# Troubleshooting

| Symptom / code | Meaning and next action |
| --- | --- |
| CAPABILITIES_PENDING / NO_ROUTE | Run `modelport doctor`. Check the probe result, installed versions and adapter restrictions. Planned integrations are not selectable. |
| DEVICE_UNAVAILABLE | This release verifies CPU only. Request CPU explicitly; CUDA is not silently substituted. |
| MISSING_ARCHITECTURE | SafeTensors contains weights, not an executable graph. Provide a registered bundle with validated signatures. |
| INVALID_WEIGHTS / INVALID_SIGNATURE | Check exact registered configuration, weight names, float32 dtype and dimensions. Python constructors/imports cannot be supplied through files. |
| NATIVE_OPERATION_FAILED / exporter errors | Read bounded job logs; verify opset, graph signatures, package lock and the trusted architecture contract. There is no automatic legacy-exporter retry. |
| UNSUPPORTED_OPERATOR | Custom domains are disabled; the CPU runtime must support actual operators. Unused optimizer-added opset declarations are reported separately. |
| CONCRETE_SHAPE_REQUIRED | Unknown non-batch dimensions cannot be guessed. Supply NPZ/NumPy tensors for prediction/validation; generated-input conversions/benchmarks require resolvable shapes. |
| NO_ELIGIBLE_OPERATORS | Dynamic INT8 needs constant-weight MatMul/Gemm. Successful quantizer return alone is insufficient; integer operators and changed weights are required. |
| VALIDATION_FAILED | Inspect every output/batch and policy. A report and failed diagnostic artifact are retained. Do not loosen tolerances just to make a job green. |
| STALE_PLAN / STALE_CHECKPOINT | Source/options/packages/adapter code changed. Create a new plan; incompatible checkpoints are not reused. |
| INTEGRITY_ERROR | Artifact bytes changed after publication. Restore the original files or reimport them as a new artifact. |
| Locked SQLite database | Keep the data directory local, avoid long external write transactions, wait for the migration lock, and use the supported single-worker arrangement. |
| RESOURCE_EXHAUSTED / WORKER_CRASHED | Check size/allocation/memory limits and host RAM. Native memory limits can terminate a child; partial output remains unpublished. |
| PERMISSIONS_FAILED | Select an application-owned directory where Windows ACLs can be restricted; do not use a shared project/system root as storage. |
| Dashboard disconnected | Start `modelport serve`, check `/health/live`, then re-enter the operator token. Credentials are deliberately not persisted across refresh. |
| npm.ps1 execution policy error | Use `npm.cmd` on Windows. No execution-policy change is needed. |
| Docker named pipe missing / cannot connect to daemon | Start Docker Desktop and wait for its Linux engine. On Windows, `start-modelport.cmd` does this automatically. Check `docker info` before rebuilding. |
| Host port already allocated | Use `start-modelport.cmd -Port 8767`, or set `MODELPORT_PORT=8767` before Compose. Open the matching URL. |
| INVALID_BATCH_SIZE | Choose the model's fixed leading batch size. Requested throughput must match the actual tensor batch. |
| REMOTE_ADDRESS_BLOCKED | Only public unsigned HTTPS URLs on port 443 are accepted. Private, mixed-DNS or credential-bearing destinations are rejected. |
| CHECKSUM_MISMATCH | Downloaded bytes differ from the supplied SHA-256. Verify the pinned upstream file; do not replace the expected hash merely to hide the error. |
| DATASET_OVERLAP / calibration validation error | Use representative calibration and genuinely separate held-out samples, then create a new plan. |
| Node/OpenBLAS allocation failure during development | Run Docker builds, full native tests and browser checks sequentially on memory-constrained hosts; check free RAM and space for the OS pagefile. |

The pinned PyTorch exporter logs missing torchvision optional operators; this project
does not need torchvision for its registered MLPs. The pinned Starlette test client
emits two deprecation warnings for httpx/AnyIO; they do not represent skipped tests.
Check [verification.md](verification.md) for observed limitations, not hypothetical support.
