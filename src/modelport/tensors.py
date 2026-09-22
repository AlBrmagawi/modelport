import math
from typing import Any

from modelport.domain import TensorSpec
from modelport.errors import ModelPortError


def validate_inputs(
    inputs: dict[str, Any], specs: list[TensorSpec], limit: int = 128 * 1024**2
) -> None:
    import numpy as np

    if set(inputs) != {spec.name for spec in specs}:
        raise ModelPortError("INVALID_INPUT", "Input names must exactly match the model signature")
    symbols: dict[str, int] = {}
    total = 0
    for spec in specs:
        array = inputs[spec.name]
        if not isinstance(array, np.ndarray) or str(array.dtype) != spec.dtype:
            raise ModelPortError(
                "INVALID_DTYPE", f"{spec.name} requires {spec.dtype}; automatic casts are disabled"
            )
        total += array.nbytes
        if total > limit or array.ndim != spec.rank:
            raise ModelPortError(
                "INVALID_SHAPE", f"{spec.name} rank or tensor byte budget is invalid"
            )
        for actual, declared in zip(array.shape, spec.dimensions, strict=True):
            if actual < 1 or (isinstance(declared, int) and actual != declared):
                raise ModelPortError(
                    "INVALID_SHAPE", f"{spec.name} dimensions do not match the signature"
                )
            if isinstance(declared, str):
                if declared in symbols and symbols[declared] != actual:
                    raise ModelPortError(
                        "INVALID_SHAPE", f"Shared dimension {declared} must agree across inputs"
                    )
                symbols[declared] = actual
                lower, upper = spec.bounds.get(declared, (1, 1_000_000))
                if not lower <= actual <= upper:
                    raise ModelPortError(
                        "INVALID_SHAPE", f"{declared} must be between {lower} and {upper}"
                    )
        if not np.isfinite(array).all():
            raise ModelPortError("INVALID_INPUT", "Nonfinite inputs are not accepted")
        if spec.value_range is not None and array.size:
            lo, hi = spec.value_range
            if array.min() < lo or array.max() > hi:
                raise ModelPortError(
                    "INVALID_INPUT", f"{spec.name} is outside its declared value range"
                )


def synthetic(specs: list[TensorSpec], batch: int, seed: int) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    result = {}
    for spec in specs:
        shape = []
        for index, dim in enumerate(spec.dimensions):
            if isinstance(dim, int):
                value = dim
            elif index == 0 and isinstance(dim, str):
                # Explicit caller-selected batch; never guess unknown dimensions.
                value = batch
            elif spec.shape is not None:
                value = spec.shape[index]
            else:
                raise ModelPortError(
                    "CONCRETE_SHAPE_REQUIRED",
                    f"Supply concrete inputs for unresolved dimensions of {spec.name}",
                )
            shape.append(value)
        dtype = np.dtype(spec.dtype)
        if math.prod(shape) * dtype.itemsize > 128 * 1024**2:
            raise ModelPortError("RESOURCE_EXHAUSTED", "Synthetic tensor exceeds allocation budget")
        if dtype.kind in "iu":
            if spec.value_range is None:
                raise ModelPortError(
                    "INPUT_RANGE_REQUIRED", "Integer synthetic inputs require explicit value ranges"
                )
            lo, hi = spec.value_range
            array = rng.integers(int(lo), int(hi) + 1, size=shape, dtype=dtype)
        elif dtype.kind == "b":
            array = rng.integers(0, 2, size=shape).astype(dtype)
        else:
            array = (
                rng.uniform(*spec.value_range, size=shape)
                if spec.value_range
                else rng.standard_normal(shape)
            ).astype(dtype)
        result[spec.name] = array
    validate_inputs(result, specs)
    return result
