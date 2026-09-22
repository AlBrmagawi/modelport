# ADR 0001: narrow verified CPU release

Accepted: 2026-09-21.

- **Stateful capability graph:** file extensions cannot distinguish execution,
  packaging, optimization and quantization. Search considers only adapters with
  successful real environment probes. Model compatibility remains a separate check.
- **Trusted registry:** SafeTensors protects the weight container, not arbitrary
  Python construction. Two bounded, versioned implementations live in application
  code. Uploaded configurations never name imports, constructors or callbacks.
- **Content-addressed bundles:** canonical path/size/SHA-256 inventories include
  external data. Staging and atomic directory rename precede database visibility.
  A crash between these systems produces an unreferenced blob, not a usable record.
- **SQLite orchestration:** one bounded local worker is enough. Attempts, durable
  events, leases, cancellation intent and ownership tokens provide explicit recovery
  without Redis/Celery. The schema is migrated by coordinated startup, not workers.
- **Subprocess execution:** ONNX parsing/export/runtime failures must not crash HTTP
  handlers. JSON/files replace Python object IPC. Windows Job Objects and POSIX limits
  support lifetime/resource controls, with platform differences documented honestly.
- **Vite + React:** the empty workspace favored a standalone thin dashboard. FastAPI
  serves its production files; no second application server is needed in local use.
- **Modern PyTorch exporter:** dynamo, opset 18, explicit `dynamic_shapes`, no legacy
  fallback. Exporter optimization folds constant weight transposes before quantization.
- **Reduced-range dynamic INT8:** the host lacks AVX512-VNNI; U8S8 saturation caused
  a real validation failure. Explicit `reduce_range=True` corrected it while keeping
  the original numerical policy. This option is recorded in plans and manifests.
- **Explicit public remote imports:** use HTTP Core's network backend to pin and
  verify socket destinations while keeping the original hostname for TLS. Each
  redirect is revalidated; no ambient credentials/proxies; full Hub revisions and
  hashes are required. Actual downloads are verified separately from offline probes.
