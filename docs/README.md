# ModelPort documentation

Start with [installation and first run](getting-started.md), then run the
[offline demo](demo.md). Return to the [project overview](../README.md).

## Use the application

- [Installation, prerequisite downloads and first-run checks](getting-started.md)
- [Libraries, lockfiles and dependency installation](dependencies.md)
- [Supported formats and capability matrix](capability-matrix.md)
- [Registered model-bundle format](model-bundle.md)
- [Calibration, held-out datasets and precision](datasets.md)
- [HTTPS and Hugging Face imports](remote-imports.md)
- [Numerical validation policies](validation.md)
- [Benchmark methodology and reports](benchmarking.md)
- [Troubleshooting](troubleshooting.md)

## Operate and extend

- [Deployment, configuration, backup and restore](deployment.md)
- [Security boundary](../SECURITY.md)
- [Architecture and conversion flow](architecture.md)
- [Manifests and provenance](manifests.md)
- [Adapter authoring](adapter-authoring.md)
- [Core design decision](adr/0001-core-design.md)
- [Contributor workflow](../CONTRIBUTING.md)

## Release and verification

- [Release notes and wheel installation](release-notes.md)
- [Pre-push quality review](pre-push-review.md)
- [Detailed verification evidence](verification.md)
- [Tested environment](environment.md)
- [Changelog](../CHANGELOG.md)
- [Roadmap and unsupported integrations](roadmap.md)

The running application serves its API reference at `/docs` and its generated
OpenAPI contract at `/openapi.json`. The checked-in contract is [openapi.json](openapi.json).
