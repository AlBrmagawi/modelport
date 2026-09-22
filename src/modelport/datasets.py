"""Bounded tensor datasets, with explicit calibration/validation separation."""

import hashlib
from pathlib import Path
from typing import Any

from modelport.domain import TensorDataset, TensorSpec
from modelport.errors import ModelPortError
from modelport.security import digest_file, identity, load_npz
from modelport.tensors import validate_inputs


def inspect_dataset(path: Path, purpose: str) -> TensorDataset:
    import numpy as np

    arrays = load_npz(path)
    if not 1 <= len(arrays) <= 8 or any(value.ndim == 0 for value in arrays.values()):
        raise ModelPortError("INVALID_DATASET", "Use 1..8 named tensors with a sample dimension")
    counts = {value.shape[0] for value in arrays.values()}
    if len(counts) != 1 or not 1 <= next(iter(counts)) <= 1024:
        raise ModelPortError("INVALID_DATASET", "All tensors need the same 1..1024 sample count")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ModelPortError("INVALID_DATASET", "Dataset contains nonfinite values")
    count = next(iter(counts))
    specs = [
        TensorSpec.model_validate(
            {"name": name, "dtype": str(value.dtype), "dimensions": list(value.shape)}
        )
        for name, value in sorted(arrays.items())
    ]
    # Sample identities catch overlap even when NPZ compression, order or names of files differ.
    sample_digests = [
        identity(
            {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape[1:]),
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(value[index]).tobytes()
                    ).hexdigest(),
                }
                for name, value in sorted(arrays.items())
            }
        )
        for index in range(count)
    ]
    digest = digest_file(path)
    return TensorDataset.model_validate(
        {
            "dataset_id": identity({"sha256": digest, "purpose": purpose}),
            "sha256": digest,
            "logical_digest": identity(sample_digests),
            "purpose": purpose,
            "sample_count": count,
            "sample_digests": sample_digests,
            "tensors": [spec.model_dump() for spec in specs],
            "size_bytes": path.stat().st_size,
        }
    )


def calibration_batches(
    path: Path, specs: list[TensorSpec], batch_size: int
) -> list[dict[str, Any]]:
    arrays = load_npz(path)
    count = next(iter(arrays.values())).shape[0]
    result = []
    for start in range(0, count, batch_size):
        batch = {name: value[start : start + batch_size] for name, value in arrays.items()}
        validate_inputs(batch, specs)
        result.append(batch)
    return result


def require_independent(calibration: TensorDataset, validation: TensorDataset) -> None:
    if calibration.purpose != "calibration" or validation.purpose != "validation":
        raise ModelPortError(
            "DATASET_PURPOSE",
            "Choose calibration and validation datasets with their declared roles",
        )
    if calibration.sample_count < 8:
        raise ModelPortError(
            "INVALID_DATASET", "Static INT8 requires at least eight calibration samples"
        )
    if set(calibration.sample_digests).intersection(validation.sample_digests):
        raise ModelPortError(
            "DATASET_OVERLAP", "Calibration and validation must not contain identical samples"
        )
