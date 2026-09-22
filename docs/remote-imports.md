# Public HTTPS and Hugging Face imports

Importing a repository does not imply support for its architecture. This release
accepts explicit ONNX/JSON/SafeTensors/external-data file lists, then applies the
same inspection as local import. It never clones a repository, executes its Python,
enables `trust_remote_code`, loads pickle or reads an ambient Hugging Face token.

```sh
modelport import-hub examples/hub-mnist.json
modelport import-hub examples/hub-tiny-bert.json
python scripts/verify_remote.py --data-dir .tools/remote-verification
```

The last command is an opt-in live test. It downloads public pinned MNIST and tiny
random BERT ONNX files, verifies SHA-256 hashes, runs finite CPU predictions and
checks that direct HTTPS imports produce identical artifact IDs. BERT uses explicit
int64 `input_ids`, `attention_mask` and `token_type_ids`, each shape `[2,4]`; its output
is `[2,4,32]`. This verifies an existing BERT ONNX graph, not a generic Transformers
SafeTensors loader or tokenizer. The tiny random weights have no useful task accuracy.

The dashboard's **Import model → Source** selector accepts local files, public HTTPS
or Hugging Face JSON manifests. The CLI uses `import-https FILE.json` or
`import-hub FILE.json`. REST endpoints are `POST /api/v1/imports/https` and
`POST /api/v1/imports/huggingface`; all require the operator token and return jobs.

```json
{"files":[{"url":"https://example.org/model.onnx","path":"model.onnx","sha256":"64 lowercase hex characters"}]}
```

Hub manifests require `repository`, full 40-character `revision`, and `files` with
repository `path` and `sha256`. Optional `destination` moves an explicitly selected
nested ONNX graph to the bundle root. Preserve its external-data relative paths
when listing companion files. See the two checked-in example manifests for exact
verified public references and hashes; no model weights ship in source archives.

Only public HTTPS port 443 is permitted. Each hop resolves only public unicast
addresses; actual connections pin and check numeric peers while TLS verifies the
original host. At most three redirects are followed. Proxies, authorization, cookies,
private IPs, initial URL query credentials and compressed HTTP payloads are disabled.
All files need predeclared SHA-256 hashes and obey 128 MiB/file, 256 MiB/bundle,
64-file and job-time budgets. Public CDN redirects may contain signed query strings;
those redirect URLs are never persisted. Private or gated repositories are unsupported.

Upstream examples: [MNIST](https://huggingface.co/onnxmodelzoo/mnist-8),
[tiny BERT](https://huggingface.co/hf-internal-testing/tiny-random-BertModel).
Review upstream model terms before use; ModelPort's license does not cover weights.
