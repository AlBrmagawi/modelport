# Reproducible demonstration

```sh
uv run --no-sync modelport demo --output ./demo-results
uv run --no-sync modelport demo --architecture dual-input --output ./demo-dual-results
```

The first command generates seeded registered weights, imports real bundle bytes,
exports FP32 ONNX, validates it, creates separately persisted optimized and dynamic
INT8 variants, validates both against the original source, benchmarks all four
representations sequentially, and loads/predicts through the SDK. No external model
repository is contacted. A separate NPZ dataset uses seed 4242; initial conversion
checks use seed 2027 and concrete batch sizes 2 and 4.

Outputs: `summary.json`, `capabilities.json`, `validation-inputs.npz`,
`validations.json`, `benchmarks.json`, `benchmarks.csv`, `benchmarks.html`, immutable
manifest copies and downloadable artifact ZIPs. The self-contained HTML report
includes actual measurements and full configuration. The demo data is synthetic;
it says nothing about real-world task accuracy.

With `serve` on the same data directory, records appear immediately in the dashboard.
The dashboard's **Run CPU demo** button enqueues actual fixture, conversion,
validation and benchmark jobs; it never seeds fictional result rows.

The second architecture accepts `left` and `right` arrays and returns `logits` and
`features`. [examples/multiple_inputs.py](../examples/multiple_inputs.py) executes
batch sizes 2 and 7 and verifies that inconsistent shared dimensions are rejected.
[examples/failure.py](../examples/failure.py) demonstrates actionable device rejection
and a deliberately perturbed output failing the numerical gate.

The browser test uploads the actual generated bundle, selects a backend-generated
plan, waits for conversion and validation, measures inference, downloads a report
and an artifact, reloads persisted history, checks mobile overflow, and runs an axe
WCAG 2 A/AA check. Screenshots in `docs/screenshots/` are captured from this real run.
