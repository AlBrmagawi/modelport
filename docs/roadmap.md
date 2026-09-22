# Roadmap

The verified release is a local CPU application. This roadmap separates shipped
work from future possibilities; it does not promise delivery dates.

## Implemented and verified locally

- Registered PyTorch + SafeTensors and standard-domain ONNX imports.
- Real PyTorch → ONNX export and ONNX Runtime CPU execution.
- ONNX optimization, dynamic INT8, calibrated static INT8 and FP16 graph conversion.
- Numerical validation, actual benchmarks and exportable evidence.
- Persistent jobs, cancellation, interruption recovery and verified checkpoints.
- Python SDK, CLI, authenticated API and React dashboard.
- Checksummed public HTTPS and commit-pinned Hub imports, including tiny BERT ONNX.
- Windows/Linux CPU tests, browser workflow, Docker and installed-wheel checks.

See [verification](verification.md) and the live GitHub Actions status for evidence.

## Next useful improvements

- More trusted architecture definitions and explicit model-family examples.
- Broader operator coverage with per-model compatibility diagnostics.
- More real-world calibration and task-evaluation examples.
- Workspace retention controls and richer report comparisons.
- A reproducible macOS verification environment.

## Future runtime adapters

OpenVINO, TensorFlow/LiteRT, Core ML, TensorRT, GGUF and GPU execution have no
verified executable route in this release. An adapter must implement actual import,
conversion, inference, validation and documentation gates before it becomes selectable.
See [adapter authoring](adapter-authoring.md).

Generic Transformers reconstruction, distributed workers and public multi-tenant
hosting also remain outside the implemented scope. Propose a focused change through
a [feature request](https://github.com/AlBrmagawi/modelport/issues/new/choose).
