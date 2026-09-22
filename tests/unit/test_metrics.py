import json

import numpy as np
import pytest

from modelport.domain import TensorSpec
from modelport.errors import ModelPortError
from modelport.tensors import synthetic, validate_inputs
from modelport.validation import compare_arrays, policy_named


def test_corruption_is_not_hidden_by_averages():
    source = np.zeros((1000,), dtype=np.float32)
    target = source.copy()
    target[-1] = 0.1
    report = compare_arrays(source, target, policy_named("fp32-default"))
    assert not report["passed"]
    assert report["max_absolute_error"] > 0.09


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_fails_without_nonstandard_json(value):
    report = compare_arrays(
        np.array([1], dtype=np.float32),
        np.array([value], dtype=np.float32),
        policy_named("fp32-default"),
    )
    assert not report["passed"]
    json.dumps(report, allow_nan=False)


def test_zero_empty_dtype_integer_and_shape_semantics():
    policy = policy_named("fp32-default")
    zeros = np.zeros((2, 3), dtype=np.float32)
    result = compare_arrays(zeros, zeros.copy(), policy)
    assert result["passed"] and result["cosine_similarity"] is None
    assert not compare_arrays(zeros, zeros.astype(np.float64), policy)["passed"]
    assert not compare_arrays(zeros, zeros[:, :2], policy)["passed"]
    assert not compare_arrays(zeros[:0], zeros[:0], policy)["passed"]
    assert not compare_arrays(np.array([1]), np.array([2]), policy)["passed"]
    assert compare_arrays(np.array([True]), np.array([True]), policy)["passed"]


def test_extreme_finite_outputs_do_not_emit_infinite_metrics():
    result = compare_arrays(np.array([1e308]), np.array([-1e308]), policy_named("fp32-default"))
    assert not result["passed"] and result["normalized_l2"] is None
    json.dumps(result, allow_nan=False)


def test_synthetic_unknown_dimensions_never_become_one():
    spec = TensorSpec(name="x", dtype="float32", dimensions=["batch", "unknown"])
    with pytest.raises(ModelPortError, match="concrete"):
        synthetic([spec], 2, 1)


def test_shared_symbols_dtype_and_integer_range():
    specs = [
        TensorSpec(name=n, dtype="float32", dimensions=["batch", 4], bounds={"batch": (1, 64)})
        for n in ("a", "b")
    ]
    with pytest.raises(ModelPortError, match="Shared dimension"):
        validate_inputs(
            {"a": np.zeros((2, 4), np.float32), "b": np.zeros((3, 4), np.float32)}, specs
        )
    with pytest.raises(ModelPortError, match="automatic casts"):
        validate_inputs(
            {"a": np.zeros((2, 4), np.float64), "b": np.zeros((2, 4), np.float32)}, specs
        )
    integer = TensorSpec(name="index", dtype="int64", dimensions=["batch", 2], value_range=(0, 10))
    values = synthetic([integer], 4, 3)["index"]
    assert values.min() >= 0 and values.max() <= 10
