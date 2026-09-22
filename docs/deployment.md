# Local deployment and operations

Use the [README quickstart](../README.md). `uv sync --locked --extra cpu` installs
the hash-locked dependencies using the explicit PyTorch CPU index. Other frameworks
are not installed. The base package and `--help` remain lazy with respect to ML imports.
Only the implemented CPU integration has an installable extra.

The release wheel includes the built dashboard, so installed users need no Node.
Source development builds `web/dist`; `MODELPORT_WEB_DIR` can override either path.
The source distribution includes the frontend and operational files. No UI asset
is fetched from a CDN at runtime. `MODELPORT_REQUIRE_WEB=1` makes release builds
fail if the dashboard has not been built.
`/docs` and `/redoc` serve a local endpoint reference and link to `/openapi.json`.
They need no CDN scripts and work with the same strict content security policy.

## Configuration and processes

- CLI `--data-dir PATH` / SDK `data_dir=` takes precedence over `MODELPORT_DATA_DIR`;
  default `.modelport` is relative to the launch directory.
- `MODELPORT_TOKEN` overrides the generated `operator-token` file; at least 24
  characters. `.env.example` documents variables but `.env` is not loaded implicitly.
- `MODELPORT_WEB_DIR` overrides the built dashboard path.
- `MODELPORT_ALLOWED_HOSTS` and `MODELPORT_ALLOWED_ORIGINS` are comma-separated exact
  allowlists. Defaults are loopback hosts and documented local development origins.
- Programmatic `Settings` controls resource budgets. Web requests cannot override
  arbitrary host resources, imports, device providers or subprocess commands.

`modelport serve` starts a separate supervisor process and serves the dashboard on
127.0.0.1:8765. `--no-with-worker` supports a separately managed `modelport worker`.
Run `modelport migrate` first for an external worker. Migration takes an OS file
lock; workers never run migrations themselves. Current schema head is `0003`.
`/health/live` reports process liveness; `/health/ready` checks a current worker
heartbeat. Startup schedules new probes when stored capability evidence is stale.

Ctrl+C or service shutdown requests a bounded supervisor stop. The current child
tree is terminated and its job is marked interrupted. Forced process termination
uses Job Object cleanup on Windows; stale leases are marked interrupted after 15
seconds. Retry preserves attempt history and reuses only rehashed compatible step
checkpoints. A running exporter cannot resume at an arbitrary instruction.

Remote binding requires `--host`, explicit host/origin allowlists, a private network,
TLS/authentication at a trusted reverse proxy and a reviewed isolation boundary.
Changing the bind address alone does not make this deployment safe for the internet.

## Docker

```sh
docker compose -f docker/compose.yml up --build
docker compose -f docker/compose.yml exec modelport modelport token
```

The default Compose service uses CPU wheels, nonroot UID 10001, persistent `/data`,
a read-only root, temporary `/tmp`, a loopback published port, dropped capabilities,
no-new-privileges, process/memory/CPU limits, an init process and readiness checking.
Base images are pinned to the digests used in verification. CPU dependencies occupy
a separate cached build layer; changing application code does not redownload Torch.
`MODELPORT_PORT=8767` selects an alternate host port and matching browser origins.
`docker/compose.isolated.yml` is an
optional **diagnostic native runner**, separate from durable queue orchestration:
network disabled, root filesystem and input mount read-only, 512 MiB working tmpfs,
process/memory limits, nonroot UID and dropped capabilities. It returns candidate
results; it does not bypass the supervisor to publish database artifacts.

Prepare a real benchmark request with
`python scripts/prepare_container_job.py ARTIFACT_ID`. The request uses
`/inputs/ARTIFACT_ID`, and the Compose profile mounts `.modelport/blobs` at `/inputs`.
Set `MODELPORT_CONTAINER_INPUTS`, `MODELPORT_CONTAINER_REQUEST` and
`MODELPORT_CONTAINER_RESULTS` for other paths. The results directory must be writable
by container UID 10001; an administrator must set that ownership on Linux. Then run
`docker compose -f docker/compose.isolated.yml run --rm native`. Results appear in
`container-job/results/result.json`. The actual restricted profile passed a
30-sample ONNX CPU benchmark on this host, with networking disabled and inputs
read-only. Check the result JSON `ok` field; container exit alone is not success.

Docker Desktop was installed with its Linux engine stopped. Starting Desktop fixed
the missing named-pipe error; Docker 29.6.2 now builds and runs the actual CPU app.
The Windows launcher starts Desktop automatically when needed. Existing unrelated
Docker projects are left intact. See [verification](verification.md) for Linux results.

## Backup, restore and retention

Stop API and workers and close SDK sessions before a filesystem backup. Copy the
**entire data directory**, including `modelport.db`, WAL/SHM if present, `blobs/`,
`checkpoints/`, `datasets/`, staged retry inputs and the operator-token. Secure the backup like
the originals. For live backup, use SQLite's backup API and a coordinated artifact
snapshot; an arbitrary live file copy is not transactionally consistent.

Restore to an application-owned directory, restrict permissions, run migrations,
then `doctor`. Never copy a live SQLite database to a network share and treat it
as a distributed queue. Test restore by inspecting artifacts and executing a small
prediction. Model files are verified against their stored digest before execution.

`modelport gc` is a dry run. `--apply` removes eligible unreferenced application-owned
blobs and staging older than the selected retention window, while holding the worker
lock and refusing pending jobs. Records, referenced source inputs and reports are
retained for reproducibility/retry. Logs are capped at 256 KiB/job. Finished worker
directories are removed; forced-crash work directories may need offline cleanup
after recovery. There is no automatic model/history deletion in this release.

Expired uploads cannot be submitted after 24 hours. Staged inputs already referenced
by jobs are retained. Disk quotas and large-scale history pruning are future operations
work; the local operator controls the data directory.
