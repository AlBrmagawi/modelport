# Capabilities and verified routes

Verification host and exact versions are in [environment.md](environment.md).

| Adapter / operation | Implementation | Environment verification | Model contract |
| --- | --- | --- | --- |
| Local file/tree and multipart import | Implemented | Passed | 64 files, 128 MiB each, 256 MiB total; no archives |
| SafeTensors inspection and registered binding | Implemented | Passed | Valid header, offsets, names/shapes/dtypes; bundle.json |
| PyTorch bundle → ONNX FP32 | Implemented | CPU export + inference + numerical checks passed | Registry v1, float32, opset 18, batch 1–64 |
| ONNX → basic optimized ONNX | Implemented | Graph changed from 6 to 4 nodes on fixture; execution passed | Standard operator domains; no-op explicitly reported |
| ONNX → dynamic INT8 ONNX | Implemented | Two integer MatMul nodes, INT8 weights, real inference and validation passed | Constant-weight MatMul/Gemm, QInt8 reduced range, no calibration |
| PyTorch / ORT unified loading | Implemented | Batch 1 and 4; dual input batch 2 and 7; bad shapes rejected | Named NumPy tensors, CPU only, one caller per loaded handle |
| Independent validation / benchmark / reports | Implemented | Actual runtime outputs and timer samples | Recorded inputs/policy/runtime; no universal accuracy or speed guarantee |
| HTTPS, Hugging Face | Implemented import operations | Actual MNIST and tiny BERT downloads and CPU predictions passed | Public HTTPS, pinned DNS/socket peers, SHA-256, full Hub commit; no credentials |
| ONNX static INT8 | Implemented | Actual QDQ matrix weights and original-source validation passed | Separate calibration/validation NPZ; no overlapping samples; MatMul/Gemm |
| ONNX FP16 | Implemented | Actual FP16 weights, ORT CPU prediction and validation passed | FP32 external inputs/outputs; CPU arithmetic may promote to FP32 |
| OpenVINO, Core ML, TensorRT, TensorFlow/LiteRT, GGUF | Planned, no executable edge | Not installed or executed | Separate architecture/platform/device contracts required |

`doctor` enforces declared package version constraints before native probes, and reports
installed packages, advertised ORT providers, and independently executed adapter probes.
Supported FP16 and static/dynamic INT8 graphs retain distinct precision labels.
Other mixed/unsupported precisions cannot enter FP32 transformation routes.
The installed ORT package advertises Azure and CPU providers;
only **CPUExecutionProvider** is selected or claimed. The RTX 2060 hardware listing
is not evidence of GPU model support. `device="cuda"` fails; fallback is never implicit.

The first release does not infer trainable parameter counts from ONNX initializers.
It labels initializer-element counts explicitly. Weight-only files remain nonexecutable.
No ONNX-to-PyTorch, ONNX-to-GGUF, arbitrary SafeTensors conversion, or dequantization
route is invented.
