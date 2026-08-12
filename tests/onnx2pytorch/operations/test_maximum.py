import numpy as np
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.mark.parametrize("op_type", ["Max", "Min"])
@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_variadic(op_type, count):
    """These used to map onto torch.max and torch.min, which reduce a single
    tensor to its extreme value and reject more than two."""
    np.random.seed(0)
    inputs = {
        "in{}".format(i): np.random.randn(2, 3).astype(np.float32) for i in range(count)
    }
    model = make_single_node_model(op_type, inputs, 13)
    assert_matches_oracle(model, inputs)


@pytest.mark.parametrize("op_type", ["Max", "Min"])
def test_broadcast(op_type):
    np.random.seed(0)
    inputs = {
        "a": np.random.randn(2, 1, 4).astype(np.float32),
        "b": np.random.randn(3, 1).astype(np.float32),
        "c": np.random.randn(4).astype(np.float32),
    }
    model = make_single_node_model(op_type, inputs, 13)
    assert_matches_oracle(model, inputs)


@pytest.mark.parametrize("op_type", ["Max", "Min"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float64])
def test_dtypes(op_type, dtype):
    inputs = {
        "a": np.array([1, 5, 2], dtype=dtype),
        "b": np.array([3, 1, 4], dtype=dtype),
    }
    model = make_single_node_model(op_type, inputs, 13)
    assert_matches_oracle(model, inputs)
