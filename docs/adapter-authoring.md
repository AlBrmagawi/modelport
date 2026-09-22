# Adding an adapter

Adapters are trusted application code. This release has a closed administrator-owned
registry in `planning.py` and `architectures.py`; it never discovers plugins declared
by model files and never installs uploaded dependencies.

A runtime follows this concrete structural contract from `ports.py`:

```python
class Runtime:
    info: dict  # runtime version, requested/actual device, providers, threads

    def predict(self, inputs: dict):
        # Validate exact names, dtypes, dimensions and budgets before native calls.
        # Return an explicit mapping of output names to NumPy arrays.
        ...

    def close(self) -> None:
        # Release native sessions, buffers and subprocess ownership.
        ...
```

`native.NativeModel` is the implemented example. A new converter must declare stable
ID/version, source/target state predicates, an option schema, package bounds,
architecture/operator/shape/device restrictions, required inputs/calibration/reference,
fidelity/portability warnings, preflight checks and an output verification contract.
The existing `torch-onnx` edge is a concrete model to follow, not a license to claim
arbitrary graph export compatibility.

Implement the native entry point in a supervised child using `ExecutionContext`.
Only return output candidates from its private work root. The supervisor rehashes
and publishes; adapters do not write usable artifact records. Add a real environment
probe, then model-specific integration tests and numerical checks. Until those pass,
list the integration as planned or experimental and keep it ineligible for search.

Changes to trusted architecture code or transformation behavior require a version
change and invalidate environment/plan identities. Never infer a universal route
from two matching file extensions. New runtimes may require separate interfaces,
such as token generation, instead of forcing everything through tensor prediction.
