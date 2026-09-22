# Calibration, validation and precision

Datasets are NPZ files containing 1–8 named numeric arrays. Names and dtypes must
match the model exactly. The leading dimension contains 1–1024 examples and must
agree across arrays. Object arrays, nonfinite values, excessive allocations and
unsafe ZIP entries are rejected. Inspection runs in a bounded native worker.

```python
calibration = client.import_dataset("calibration.npz", purpose="calibration")
validation = client.import_dataset("held-out.npz", purpose="validation")
plan = client.plan(fp32.artifact_id, precision="static-int8",
    calibration_id=calibration.dataset_id, validation_id=validation.dataset_id,
    calibration_batch_size=4)
quantized = client.convert(plan)
```

Use representative data for calibration and a separate held-out dataset for
validation. At least eight calibration examples are required. ModelPort computes
per-sample identities across all named tensors and rejects any overlap. File hashes,
logical identities, sample counts, tensor signatures and purposes enter the plan.
Both datasets are rehashed before execution. A different NPZ encoding cannot hide
identical samples. This protects sample separation; it does not establish dataset
representativeness, real task accuracy, or absence of correlated examples.

The CLI exposes `modelport datasets import FILE --purpose calibration` and
`modelport datasets list`. `modelport plan --help` lists dataset ID flags. The
dashboard exposes uploads and selectors when **Static INT8** is selected. The API
accepts multipart `POST /api/v1/datasets?purpose=calibration` and returns an inspection
job. `validation` is the other purpose. IDs can also be passed to `/api/v1/validations`.

Static INT8 uses ORT MinMax calibration, QDQ, unsigned activations, reduced-range
signed weights and supported constant-weight MatMul/Gemm. A real quantized matrix
and QDQ nodes must be present. Every required output is validated against the original
source using normalized L2 <= 0.05. The seeded fixture passed with 512 calibration
examples and an independent four-example validation set. An earlier 64-example set
failed; the policy was not loosened. A failed conversion keeps diagnostic evidence
and cannot be loaded normally as validated.

FP16 uses `precision="fp16"`, preserves FP32 inputs/outputs, requires actual FP16
initializers, sorts the converted graph and checks it with ONNX and ORT. Its
`fp16-default` policy gates normalized L2 <= 0.005. CPU execution can promote kernels
to FP32; no FP16 arithmetic or performance improvement is promised. Control-flow
subgraphs are outside this adapter's supported contract.

Both transformations are real probed graph edges. Revalidation retains the target's
policy and registered held-out dataset. Files are retained for reproducibility;
automatic dataset/history deletion is not implemented.
