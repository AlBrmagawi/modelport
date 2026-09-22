# GitHub publication review

Review date: 2026-09-22. Scope: the ModelPort 0.1.0 local CPU application,
downloadable source, Python wheel and Docker setup. Checks were performed locally;
GitHub-hosted Actions had not run at the time of this local review. No commit or
push had yet been created. Current hosted results are available in
[GitHub Actions](https://github.com/AlBrmagawi/modelport/actions/workflows/ci.yml).

## Issues corrected

- A malformed non-ASCII bearer token returned HTTP 500. Authentication now performs
  its constant-time comparison on bytes and returns HTTP 401. The real API test
  includes the malformed header and passes after the fix.
- The README assumed a pre-existing environment at a machine-specific path. Its
  Windows quickstart now starts from a new checkout and creates the environment.
- Source distributions omitted contributor instructions, the Python version file
  and the frontend formatting exclusions. They are now included.
- Git, Docker and distribution exclusions now cover environment-file variants,
  operator-token files, ModelPort databases and private-key files. The configuration
  example remains included.
- CI now verifies archive checksums, wheel records, required files, license metadata,
  bundled dashboard files and correspondence with the current source. It installs
  the built wheel and exercises its actual CPU probes.
- CI secret scanning now verifies the downloaded Gitleaks archive's SHA-256 and
  scans Git history as well as the current tree. Formatting/linting also cover the
  package build hook.
- The opt-in live Docker browser check now uses the current library size instead
  of assuming exactly four models. Its screenshots go to ignored local evidence,
  keeping the operator's workspace out of publishable documentation captures.

## Executed checks

| Check | Result |
| --- | --- |
| Windows Python suite | 106 passed, one Linux-only skip; 178.23 seconds |
| Linux Python suite | 106 passed, one Windows-only skip; 179.12 seconds; nonroot, offline, read-only container |
| Malformed-token regression after the fix | Passed; invalid credentials return 401 |
| Ruff lint and formatting | Passed across application, tests, examples, scripts and build hook |
| mypy | Passed for Windows and Linux targets; 33 application files |
| Backend and generated frontend schemas | No drift |
| Frontend installation, unit tests and production build | Passed; three unit tests |
| Prettier | Passed |
| Real browser workflow | Passed in 45.5 seconds; conversion, validation, benchmarks, downloads, persistence, static INT8, remote policy, accessibility and mobile layout |
| Dependency audits | No known vulnerabilities in the installed-package, normalized Torch and npm scans |
| Gitleaks directory scan | No detected secrets |
| GitHub Actions syntax | actionlint 1.7.12 passed; downloaded tool checksum verified |
| Dependency consistency | Development and separate runtime environments both pass `uv pip check` |
| Docker runtime and test image builds | Passed |
| Offline Docker demo | Three validated conversions, four measured benchmarks and SDK prediction shape [4,10] |
| Installed release wheel | Separate runtime environment outside the checkout serves bundled UI/notices and passes all five CPU probes |
| Updated Docker app and restart | Healthy on port 8765; all eight existing model records retained; five CPU probes; malformed token returns 401 |
| Live Docker browser | Current eight-model library, inspection, five CPU probes and mobile layout pass; no JavaScript errors |
| Source archive setup | Extracted Compose files parse and the locked Python dependencies remain consistent |
| Distribution integrity | Eight checksum entries verified; source and wheel match the workspace; wheel records, licenses, metadata and private-file exclusions pass |

Two Python warnings come from upstream Starlette/AnyIO deprecations. They are not
suppressed. The installed-package advisory service cannot identify unpublished
ModelPort or the `+cpu` Torch version; the script separately audits upstream
Torch 2.13.0. These are point-in-time dependency scans, not an audit of every
operating-system package in the container.

The core tests use actual generated PyTorch and ONNX models. The normal test suite
does not download models. Live public MNIST/BERT downloads were verified in the
earlier release audit and are not prerequisites for this pre-push check.

## Files intended for GitHub

The prospective Git file list was inspected using an isolated temporary Git directory.
Ignored files include local credentials, `.modelport`, `.tools`, `.venv`, generated
models, `node_modules`, release archives and browser diagnostics. All eleven
sensitive-path exclusion probes passed, `.env.example` remains visible, and local
Markdown links resolve. No existing repository history was available to scan.

The wheel and source archive are rebuilt under `release/0.1.0/`. Keep this directory
out of the source commit; attach its files to a separately reviewed release when
you choose to publish downloadable packages. Use `SHA256SUMS.txt` to verify them.

## Reproduce and inspect

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the command sequence. Local evidence:

- `.tools/pytest.xml`: Windows test results.
- `.tools/docker-evidence/`: Docker/Linux evidence.
- `.tools/pip-audit.json`, `.tools/torch-audit.json`, `.tools/gitleaks-report.json`.
- `.tools/release-verification.json`: archive and wheel integrity results.
- `.tools/github-files-review.json`: prospective source-file inventory.
- `web/test-results/`: real browser run.

After the first push, check the Windows, Ubuntu, secrets and container jobs in
GitHub Actions before tagging the public release. GPU, macOS and optional runtime
adapters remain outside the verified release scope described in the
[capability matrix](capability-matrix.md).
