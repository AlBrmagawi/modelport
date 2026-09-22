"""Explicit trusted registry. Uploaded files never choose Python imports."""

import math
from pathlib import Path
from typing import Any, Literal, cast

from modelport.config import Settings
from modelport.domain import ArchitectureConfig, ModelBundleSpec, TensorSpec
from modelport.errors import ModelPortError
from modelport.security import digest_file, read_json, safetensors_header, write_json

REGISTRY_VERSION = "1"


def signatures(
    architecture: str, config: ArchitectureConfig
) -> tuple[list[TensorSpec], list[TensorSpec]]:
    common: dict[str, Any] = {"dtype": "float32", "bounds": {"batch": (1, 64)}}
    if architecture == "vision-mlp":
        side = math.isqrt(config.features)
        if side * side != config.features:
            raise ModelPortError("INVALID_CONFIG", "vision-mlp features must be a perfect square")
        inputs = [TensorSpec(name="images", dimensions=["batch", 1, side, side], **common)]
        outputs = [
            TensorSpec(name="logits", dimensions=["batch", config.classes], class_axis=-1, **common)
        ]
    elif architecture == "dual-input":
        inputs = [
            TensorSpec(name=n, dimensions=["batch", config.features], **common)
            for n in ("left", "right")
        ]
        outputs = [
            TensorSpec(
                name="logits", dimensions=["batch", config.classes], class_axis=-1, **common
            ),
            TensorSpec(name="features", dimensions=["batch", config.hidden], **common),
        ]
    else:
        raise ModelPortError("MISSING_ARCHITECTURE", "Architecture is not in the trusted registry")
    return inputs, outputs


def construct(spec: ModelBundleSpec):
    import torch

    expected_inputs, expected_outputs = signatures(spec.architecture, spec.config)
    if spec.inputs != expected_inputs or spec.outputs != expected_outputs:
        raise ModelPortError(
            "INVALID_SIGNATURE",
            "Bundle signature does not match the registered architecture contract",
        )
    config = spec.config

    class VisionMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(config.features, config.hidden)
            self.last = torch.nn.Linear(config.hidden, config.classes)

        def forward(self, images):
            # Explicit MatMul + Add keeps the dynamic quantizer's eligible operation clear.
            hidden = torch.relu(images.flatten(1) @ self.first.weight.T + self.first.bias)
            return hidden @ self.last.weight.T + self.last.bias

    class DualInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(config.features, config.hidden)
            self.last = torch.nn.Linear(config.hidden, config.classes)

        def forward(self, left, right):
            hidden = torch.relu((left + right) @ self.first.weight.T + self.first.bias)
            return hidden @ self.last.weight.T + self.last.bias, hidden

    return (VisionMLP() if spec.architecture == "vision-mlp" else DualInput()).eval()


def load_bundle(directory: Path, settings: Settings):
    import torch
    from safetensors.torch import load_file

    spec = ModelBundleSpec.model_validate(read_json(directory / "bundle.json"))
    header = safetensors_header(
        directory / spec.weights, settings.max_tensor_bytes, settings.max_metadata_bytes
    )
    for filename, expected in spec.expected_digests.items():
        if filename != spec.weights or digest_file(directory / filename) != expected:
            raise ModelPortError(
                "INTEGRITY_ERROR", "Supplied expected weight digest does not match"
            )
    model = construct(spec)
    expected_state = model.state_dict()
    actual = {t["name"]: t for t in header["tensors"]}
    if set(actual) != set(expected_state):
        raise ModelPortError(
            "INVALID_WEIGHTS", "Weight names do not match the trusted architecture"
        )
    for name, tensor in expected_state.items():
        if actual[name]["shape"] != list(tensor.shape) or actual[name]["dtype"] != "F32":
            raise ModelPortError(
                "INVALID_WEIGHTS", "Weight shapes/dtypes do not match the registered architecture"
            )
    weights = load_file(str(directory / spec.weights), device="cpu")
    if any(not torch.isfinite(value).all().item() for value in weights.values()):
        raise ModelPortError("INVALID_WEIGHTS", "Nonfinite weights are rejected")
    model.load_state_dict(weights, strict=True)
    return model.eval(), spec, header


def create_fixture(
    directory: Path, architecture: str = "vision-mlp", seed: int = 17
) -> ModelBundleSpec:
    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    config = ArchitectureConfig()
    inputs, outputs = signatures(architecture, config)
    spec = ModelBundleSpec(
        architecture=cast(Literal["vision-mlp", "dual-input"], architecture),
        config=config,
        inputs=inputs,
        outputs=outputs,
        task="synthetic fixture; not a trained classifier",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = construct(spec)
    save_file(
        model.state_dict(),
        str(directory / spec.weights),
        metadata={"seed": str(seed), "scope": "synthetic"},
    )
    spec.expected_digests = {spec.weights: digest_file(directory / spec.weights)}
    write_json(directory / "bundle.json", spec.model_dump(mode="json"))
    return spec
