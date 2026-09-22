# PyTorch bundle v1

A bundle is a directory with `bundle.json` and `weights.safetensors`. Uploaded
bundles use those two separate files in the multipart `files` fields. Archives are
export-only: extraction is intentionally not part of importing.

```json
{
  "schema_version": 1,
  "architecture": "vision-mlp",
  "architecture_version": "1",
  "config": {"features": 64, "hidden": 128, "classes": 10},
  "weights": "weights.safetensors",
  "inputs": [{"name": "images", "dtype": "float32", "dimensions": ["batch", 1, 8, 8], "bounds": {"batch": [1, 64]}}],
  "outputs": [{"name": "logits", "dtype": "float32", "dimensions": ["batch", 10], "bounds": {"batch": [1, 64]}, "class_axis": -1}],
  "preprocessing": "identity"
}
```

Constructor bounds: features 4–1024, hidden 4–1024, classes 2–1000. Vision features
must be a perfect square. Its input shape is `[batch,1,sqrt(features),sqrt(features)]`.
`dual-input` accepts `left` and `right`, each `[batch,features]`, and returns `logits`
and `features` (`[batch,hidden]`). All tensors are float32. Signatures must match the
registered architecture exactly, including symbolic relationships and batch bounds.

Weights have exactly `first.weight`, `first.bias`, `last.weight`, and `last.bias`, with
shapes determined by the constructor. Headers, offsets and allocation budgets are
checked before assigning weights. Names, shapes, dtypes and finite values must agree.
An optional `expected_digests` map may name `weights.safetensors`; supplied expected
hashes are distinguished from hashes merely observed during import.

To bind a SafeTensors file, first inspect it, then create a bundle with a known
architecture ID, validated constructor and exact signatures. Weight shapes do not
authorize guessing an executable architecture. The API accepts no Python code,
import paths, source files, serialized callables, plugins or legacy pickle payloads.

For a deterministic working example, `client.fixture("vision-mlp")` and
`client.fixture("dual-input")` generate registered bundles in a native worker. These
fixtures are small synthetic networks, not pretrained classification models.
