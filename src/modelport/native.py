"""Native CPU adapters. Import this module only inside a supervised process."""

import math
from collections import Counter
from pathlib import Path
from typing import Any

from modelport.architectures import load_bundle
from modelport.config import Settings
from modelport.domain import ModelDescriptor, ModelState, TensorSpec
from modelport.errors import ModelPortError
from modelport.security import confined, safetensors_header
from modelport.tensors import synthetic, validate_inputs


def ort_session(path: Path, threads: int = 1, optimize_to: Path | None = None):
    import onnxruntime as ort

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise ModelPortError("DEVICE_UNAVAILABLE", "CPUExecutionProvider is not available")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if optimize_to is not None:
        options.optimized_model_filepath = str(optimize_to)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def onnx_file(directory: Path) -> Path:
    files = list(directory.glob("*.onnx"))
    if len(files) != 1:
        raise ModelPortError(
            "AMBIGUOUS_BUNDLE", "An ONNX bundle must contain exactly one root ONNX graph"
        )
    return files[0]


def inspect_onnx(directory: Path, settings: Settings) -> ModelDescriptor:
    import onnx

    path = onnx_file(directory)
    model = onnx.load(str(path), load_external_data=False)
    warnings, blockers, external = [], [], []
    if model.ir_version > onnx.IR_VERSION:
        blockers.append("ONNX IR exceeds the installed checker version")
    if model.functions:
        blockers.append(
            "Local ONNX function definitions are outside the first-release operator contract"
        )
    opsets = {item.domain: item.version for item in model.opset_import}
    domains = sorted({node.domain for node in model.graph.node} | set(opsets))
    used_domains: set[str] = set()
    floating_types: set[str] = set()
    tensor_bytes = 0

    def walk(message):
        nonlocal tensor_bytes
        if isinstance(message, onnx.NodeProto):
            used_domains.add(message.domain)
        if isinstance(message, onnx.TensorProto):
            if len(message.dims) > 8 or any(d < 0 for d in message.dims):
                raise ModelPortError("INVALID_MODEL", "Invalid initializer dimensions")
            try:
                dtype = onnx.helper.tensor_dtype_to_np_dtype(message.data_type)
                if dtype.kind == "f":
                    floating_types.add(str(dtype))
                size = math.prod(message.dims) * dtype.itemsize
            except (ValueError, KeyError, TypeError) as exc:
                raise ModelPortError("UNSUPPORTED_DTYPE", "Unsupported initializer dtype") from exc
            tensor_bytes += size
            if tensor_bytes > settings.max_tensor_bytes:
                raise ModelPortError("RESOURCE_EXHAUSTED", "Initializer allocation budget exceeded")
            if message.data_location == onnx.TensorProto.EXTERNAL:
                entries = {item.key: item.value for item in message.external_data}
                if len(entries) != len(message.external_data) or set(entries) - {
                    "location",
                    "offset",
                    "length",
                    "checksum",
                }:
                    raise ModelPortError("INVALID_MODEL", "Ambiguous external tensor metadata")
                dependency = confined(directory, entries.get("location", ""))
                try:
                    offset = int(entries.get("offset", "0"))
                    length = int(entries.get("length", str(dependency.stat().st_size - offset)))
                except ValueError as exc:
                    raise ModelPortError(
                        "INVALID_MODEL", "Invalid external tensor offsets"
                    ) from exc
                if offset < 0 or length < size or offset + length > dependency.stat().st_size:
                    raise ModelPortError(
                        "INVALID_MODEL", "External tensor range is truncated or invalid"
                    )
                external.append(
                    {
                        "tensor": message.name,
                        "location": entries["location"],
                        "offset": offset,
                        "length": length,
                    }
                )
        for field, value in message.ListFields():
            if field.message_type is not None:
                for child in value if field.is_repeated else [value]:
                    walk(child)

    walk(model)
    if any(domain not in {"", "ai.onnx"} for domain in used_domains):
        blockers.append("Custom operator domains are disabled")
    if set(domains) - used_domains - {"", "ai.onnx"}:
        warnings.append(
            "Unused nonstandard opset declarations are present; no custom operators execute"
        )
    metadata_size = sum(len(x.key) + len(x.value) for x in model.metadata_props) + len(
        model.doc_string
    )
    if metadata_size > settings.max_metadata_bytes:
        raise ModelPortError("RESOURCE_EXHAUSTED", "ONNX metadata budget exceeded")
    if blockers:
        raise ModelPortError("UNSUPPORTED_OPERATOR", "; ".join(blockers))
    onnx.checker.check_model(str(path), full_check=True)
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except (onnx.shape_inference.InferenceError, RuntimeError):
        warnings.append("Shape inference could not resolve every intermediate tensor")

    def spec(value):
        if not value.type.HasField("tensor_type"):
            raise ModelPortError(
                "UNSUPPORTED_SIGNATURE", "Only named dense tensor inputs and outputs are supported"
            )
        tensor = value.type.tensor_type
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param or None
            for d in tensor.shape.dim
        ]
        dtype = str(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type))
        return TensorSpec.model_validate({"name": value.name, "dtype": dtype, "dimensions": dims})

    initializers = {value.name for value in model.graph.initializer}
    inputs = [spec(value) for value in model.graph.input if value.name not in initializers]
    outputs = [spec(value) for value in model.graph.output]
    floating_types.update(
        value.dtype for value in inputs + outputs if value.dtype.startswith("float")
    )
    operators = dict(sorted(Counter(node.op_type for node in model.graph.node).items()))
    if floating_types != {"float32"}:
        warnings.append("Graph precision is not uniformly FP32; conversion routes are restricted")
    session = ort_session(path)
    providers = session.get_providers()
    del session
    return ModelDescriptor(
        name=path.name,
        state=ModelState(
            format="onnx",
            precision="dynamic-int8"
            if any("Integer" in op or op == "DynamicQuantizeLinear" for op in operators)
            else "static-int8"
            if "QuantizeLinear" in operators
            and "DequantizeLinear" in operators
            and any(
                t.data_type in {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
                for t in model.graph.initializer
            )
            else "fp16"
            if "float16" in floating_types and floating_types <= {"float16", "float32"}
            else "fp32"
            if floating_types == {"float32"}
            else "unknown",
            opsets=opsets,
            domains=domains,
            runtime="onnxruntime",
            provider="CPUExecutionProvider",
        ),
        inputs=inputs,
        outputs=outputs,
        warnings=warnings,
        blockers=blockers,
        runtime_compatibility="compatible",
        metadata={
            "ir_version": model.ir_version,
            "operators": operators,
            "initializer_element_count": sum(
                math.prod(value.dims) for value in model.graph.initializer
            ),
            "initializer_count": len(model.graph.initializer),
            "initializer_bytes": tensor_bytes,
            "observed_floating_dtypes": sorted(floating_types),
            "external_data": external,
            "runtime_check": "session loaded; input execution checked separately",
            "providers": providers,
            "onnx_metadata": {x.key: x.value for x in model.metadata_props},
        },
    )


def inspect(directory: Path, settings: Settings) -> ModelDescriptor:
    if (directory / "bundle.json").is_file():
        if any(directory.glob("*.onnx")):
            raise ModelPortError(
                "AMBIGUOUS_BUNDLE", "Bundle contains conflicting graph representations"
            )
        model, spec, header = load_bundle(directory, settings)
        allowed = {"bundle.json", "weights.safetensors", "manifest.json"}
        if any(
            p.relative_to(directory).as_posix() not in allowed
            for p in directory.rglob("*")
            if p.is_file()
        ):
            raise ModelPortError(
                "AMBIGUOUS_BUNDLE", "PyTorch bundle includes unsupported companion files"
            )
        del model
        return ModelDescriptor(
            name=spec.architecture,
            state=ModelState(
                format="pytorch-bundle",
                precision="fp32",
                architecture=spec.architecture,
                architecture_version=spec.architecture_version,
                runtime="pytorch",
                provider="cpu",
            ),
            inputs=spec.inputs,
            outputs=spec.outputs,
            metadata={"bundle": spec.model_dump(mode="json"), **header},
            runtime_compatibility="compatible",
        )
    if any(directory.glob("*.onnx")):
        return inspect_onnx(directory, settings)
    files = list(directory.iterdir())
    weights = [p for p in files if p.suffix == ".safetensors"]
    if len(weights) == 1 and all(p.name in {weights[0].name, "manifest.json"} for p in files):
        header = safetensors_header(
            weights[0], settings.max_tensor_bytes, settings.max_metadata_bytes
        )
        return ModelDescriptor(
            name=weights[0].name,
            state=ModelState(format="safetensors"),
            metadata=header,
            blockers=[
                "Weights need a registered architecture and validated bundle.json before execution"
            ],
            runtime_compatibility="blocked",
        )
    raise ModelPortError(
        "UNSUPPORTED_FORMAT",
        "Expected one ONNX graph, one SafeTensors file, or a documented PyTorch bundle",
    )


class NativeModel:
    def __init__(
        self, directory: Path, descriptor: ModelDescriptor, settings: Settings, threads: int = 1
    ):
        self.descriptor = descriptor
        self.model: Any = None
        self.session: Any = None
        if descriptor.state.format == "pytorch-bundle":
            import torch

            torch.set_num_threads(threads)
            self.model, _, _ = load_bundle(directory, settings)
            self.info = {
                "runtime": "pytorch",
                "version": torch.__version__,
                "requested_device": "cpu",
                "actual_device": "cpu",
                "providers": ["cpu"],
                "threads": threads,
            }
        elif descriptor.state.format == "onnx":
            import onnxruntime as ort

            self.session = ort_session(onnx_file(directory), threads)
            self.info = {
                "runtime": "onnxruntime",
                "version": ort.__version__,
                "requested_device": "cpu",
                "actual_device": "cpu",
                "providers": self.session.get_providers(),
                "placement": "CPU-only session; per-node profiling not collected",
                "threads": threads,
            }
        else:
            raise ModelPortError("MISSING_ARCHITECTURE", "Weight-only files cannot be executed")

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        validate_inputs(inputs, self.descriptor.inputs)
        names = [spec.name for spec in self.descriptor.outputs]
        if self.model is not None:
            import torch

            with torch.inference_mode():
                result = self.model(
                    *(torch.from_numpy(inputs[spec.name].copy()) for spec in self.descriptor.inputs)
                )
            values = result if isinstance(result, tuple) else (result,)
            return {
                name: value.detach().cpu().numpy()
                for name, value in zip(names, values, strict=True)
            }
        return dict(zip(names, self.session.run(names, inputs), strict=True))

    def close(self) -> None:
        self.model = None
        self.session = None


def export_onnx(
    source: Path, target: Path, descriptor: ModelDescriptor, settings: Settings
) -> dict[str, Any]:
    import torch

    model, _, _ = load_bundle(source, settings)
    example = synthetic(descriptor.inputs, batch=2, seed=100)
    batch_dim = torch.export.Dim("batch", min=1, max=64)
    shapes = tuple({0: batch_dim} for _ in descriptor.inputs)
    args = tuple(torch.from_numpy(example[spec.name]) for spec in descriptor.inputs)
    target.mkdir(exist_ok=True)
    with torch.inference_mode():
        program = torch.onnx.export(
            model,
            args,
            dynamo=True,
            opset_version=18,
            verbose=False,
            dynamic_shapes=shapes,
            input_names=[s.name for s in descriptor.inputs],
            output_names=[s.name for s in descriptor.outputs],
            optimize=True,
        )
        assert program is not None
        program.save(str(target / "model.onnx"), external_data=False)
    result = inspect_onnx(target, settings)
    result.inputs, result.outputs = descriptor.inputs, descriptor.outputs
    runtime = NativeModel(target, result, settings)
    for batch in (2, 4):
        runtime.predict(synthetic(descriptor.inputs, batch=batch, seed=101 + batch))
    runtime.close()
    return {
        "exporter": "torch.export/dynamo",
        "exporter_optimize": True,
        "fallback": False,
        "opset": 18,
        "dynamic_batch": {"min": 1, "max": 64, "executed": [2, 4]},
        "torch_version": torch.__version__,
    }


def optimize_onnx(source: Path, target: Path) -> dict[str, Any]:
    import onnx

    target.mkdir(exist_ok=True)
    session = ort_session(onnx_file(source), optimize_to=target / "model.onnx")
    del session
    before = onnx.load(str(onnx_file(source)), load_external_data=False)
    after = onnx.load(str(target / "model.onnx"), load_external_data=False)
    before_nodes = [n.SerializeToString() for n in before.graph.node]
    after_nodes = [n.SerializeToString() for n in after.graph.node]
    return {
        "level": "ORT_ENABLE_BASIC",
        "nodes_before": len(before_nodes),
        "nodes_after": len(after_nodes),
        "changed": before_nodes != after_nodes,
        "no_op": before_nodes == after_nodes,
        "portability": "Standard ONNX basic rewrites; CPU provider checked",
    }


def quantize_onnx(source: Path, target: Path) -> dict[str, Any]:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target.mkdir(exist_ok=True)
    before = onnx.load(str(onnx_file(source)))
    # Use ONNX shape inference. ORT symbolic inference does not handle the modern
    # exporter's Constant value_ints form in the pinned, verified combination.
    from onnxruntime.quantization.shape_inference import quant_pre_process

    preprocessed = target.parent / "quant-preprocessed.onnx"
    quant_pre_process(
        str(onnx_file(source)), str(preprocessed), skip_optimization=False, skip_symbolic_shape=True
    )
    quantize_dynamic(
        str(preprocessed),
        str(target / "model.onnx"),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=False,
        reduce_range=True,
    )
    after = onnx.load(str(target / "model.onnx"))
    operators = Counter(n.op_type for n in after.graph.node)
    quantized = sum(operators[op] for op in ("MatMulInteger", "DynamicQuantizeMatMul"))
    if quantized == 0 or not any(
        t.data_type == onnx.TensorProto.INT8 for t in after.graph.initializer
    ):
        raise ModelPortError(
            "NO_ELIGIBLE_OPERATORS",
            (
                "Dynamic INT8 requires constant-weight MatMul/Gemm; this graph produced "
                "no quantized nodes"
            ),
        )
    return {
        "scheme": "dynamic-int8",
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": True,
        "eligible_ops": ["MatMul", "Gemm"],
        "quantized_nodes": quantized,
        "operators_before": dict(Counter(n.op_type for n in before.graph.node)),
        "operators_after": dict(operators),
        "retained_floating_ops": sorted(
            op for op in operators if op in {"Add", "Relu", "Mul", "Cast"}
        ),
    }


def quantize_static_onnx(
    source: Path, target: Path, descriptor: ModelDescriptor, calibration_path: Path, batch_size: int
) -> dict[str, Any]:
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from modelport.datasets import calibration_batches
    from modelport.security import digest_file

    batches = calibration_batches(calibration_path, descriptor.inputs, batch_size)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.iterator = iter(batches)

        def get_next(self):
            return next(self.iterator, None)

    target.mkdir(exist_ok=True)
    prepared = target.parent / "static-preprocessed.onnx"
    quant_pre_process(str(onnx_file(source)), str(prepared), skip_symbolic_shape=True)
    quantize_static(
        str(prepared),
        str(target / "model.onnx"),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
        reduce_range=True,
    )
    graph = onnx.load(target / "model.onnx")
    operators = Counter(node.op_type for node in graph.graph.node)
    quantized_weights = [
        t.name
        for t in graph.graph.initializer
        if t.data_type == onnx.TensorProto.INT8 and len(t.dims) >= 2
    ]
    if (
        not quantized_weights
        or not operators["QuantizeLinear"]
        or not operators["DequantizeLinear"]
    ):
        raise ModelPortError(
            "NO_ELIGIBLE_OPERATORS", "Static INT8 produced no quantized constant-weight MatMul/Gemm"
        )
    return {
        "scheme": "static-int8",
        "format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "method": "MinMax",
        "reduce_range": True,
        "per_channel": False,
        "quantized_weights": quantized_weights,
        "operators_after": dict(operators),
        "calibration_sha256": digest_file(calibration_path),
        "calibration_batches": len(batches),
        "calibration_samples": sum(next(iter(batch.values())).shape[0] for batch in batches),
        "preprocessing": "identity",
        "task_accuracy": None,
    }


def float16_onnx(source: Path, target: Path) -> dict[str, Any]:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    graph = onnx.load(onnx_file(source))
    if any(
        attribute.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        for node in graph.graph.node
        for attribute in node.attribute
    ):
        raise ModelPortError(
            "UNSUPPORTED_OPERATOR",
            "FP16 conversion currently requires a graph without control-flow subgraphs",
        )
    transformed = convert_float_to_float16(graph, keep_io_types=True)
    OnnxModel(transformed).topological_sort(is_deterministic=True)
    converted = [
        tensor.name
        for tensor in transformed.graph.initializer
        if tensor.data_type == onnx.TensorProto.FLOAT16
    ]
    if not converted:
        raise ModelPortError("NO_STATE_CHANGE", "The FP16 transformer did not produce FP16 weights")
    target.mkdir(exist_ok=True)
    onnx.save(transformed, target / "model.onnx")
    return {
        "scheme": "fp16",
        "keep_io_types": True,
        "fp16_initializers": converted,
        "min_positive_val": 5.96e-8,
        "max_finite_val": 65504.0,
        "runtime_note": (
            "CPU execution may promote kernels to FP32; FP16 graph/storage is verified, "
            "not native half-precision CPU arithmetic"
        ),
    }
