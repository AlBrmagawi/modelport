# Changelog

Changes are grouped by version. Git tags and GitHub Releases, when published,
identify downloadable releases; the source currently reports version 0.1.0.

## 0.1.0 — initial CPU beta

### Added

- Registered PyTorch/SafeTensors bundles and ONNX inspection.
- Modern PyTorch → ONNX export, graph optimization, dynamic/static INT8 and FP16.
- Original-source numerical validation and CPU benchmark reports.
- Durable jobs, cancellation, failure recovery, checkpoints and artifact provenance.
- Python SDK, CLI, authenticated API and React dashboard.
- Public HTTPS and pinned Hugging Face imports with explicit hashes.
- Docker packaging, Windows launcher, bundled wheel, tests and CI.
- Installation, dependency, operating and contributor documentation.

### Correctness and publication checks

- Normalized equivalent JSON numbers in conversion-plan fingerprints.
- Rejected benchmark batch mismatches and overlapping calibration/validation data.
- Preserved NPZ tensor names that overlap NumPy keyword names.
- Verified worker cleanup on Windows and Linux.
- Returned HTTP 401 for malformed non-ASCII bearer credentials.
- Added source/wheel integrity checks and exclusions for local credentials and data.

### Scope

Windows x64 and Linux x64 CPU are verified locally. The release does not claim
GPU support, arbitrary uploaded Python architectures or a public security sandbox.
