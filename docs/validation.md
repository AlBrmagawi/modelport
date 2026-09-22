# Numerical validation

Validation executes source and target on copies of the same named NumPy tensors.
Names are mapped explicitly, one-to-one and completely; dictionary iteration order
never pairs outputs. Shapes and dtypes must match. Integer/bool outputs must match
exactly. Empty outputs and NaN/Inf fail. Undefined cosine for a zero vector is null
with an explanation, never a nonstandard JSON value.

Metrics accumulate in float64: maximum/mean absolute error, maximum/mean relative
error with denominator floor `1e-8`, RMSE, normalized L2, cosine, and top-1 agreement
only for a declared class axis. Top-1 is diagnostic, not task accuracy without labels.
An exact repeat of each runtime detects observed nondeterminism.

| Versioned policy | Required gate | Scope |
| --- | --- | --- |
| fp32-default / 1 | allclose: atol 1e-5, rtol 1e-4 | Starting policy for this release's CPU float32 path |
| int8-synthetic-v1 / 1 | normalized L2 ≤ 0.05 | Explicit synthetic fixture policy; not a general production accuracy guarantee |

Every output in every batch must pass. Failed outputs are never averaged away.
The demo compares all variants to the original PyTorch source, using batches 2 and 4,
seed 2027 for the required conversion check, then an independent NPZ dataset generated
with seed 4242. Calibration is not used for dynamic quantization.

NPZ input is loaded with pickle disabled. ZIP entries and NPY headers are bounded
before tensor allocation. Each entry is one named input tensor; the NPZ is one batch.
Synthetic generation records the generator/version, seed, shapes, signature and
digest. Non-batch dynamic dimensions need explicit concrete tensors. Integer inputs
need declared ranges; shared symbols are enforced across inputs.

Reports independently identify source, target, dataset, output mapping, policy,
preprocessing, runtimes and metric implementation. `evidence_identity` hashes these
inputs. Every run is new evidence; no prior validation is silently reused. Artifact
status reflects the latest recorded validation, and the UI shows the policy and dataset
scope. `require_validated=True` requires that status. Known-failed artifacts cannot
be loaded, converted, predicted or benchmarked by default; validation can recheck a
failed target for diagnostics. Source imports begin unverified.
## Registered datasets and additional precision

Validation can use an immutable dataset ID through SDK, CLI and REST. Static INT8
requires a held-out dataset distinct from its calibration data. The calibrated INT8
policy gates normalized L2 at 0.05; FP16 gates it at 0.005. Both compare each required
output with the original source. See [datasets](datasets.md) for the exact contract.
