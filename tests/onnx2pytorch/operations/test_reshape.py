import numpy as np
import torch
import pytest

from onnx2pytorch.operations import Reshape
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.fixture
def inp():
    return torch.rand(35, 1, 200)


@pytest.fixture
def pruned_inp():
    return torch.rand(35, 1, 160)


@pytest.mark.parametrize("enable_pruning", [True, False])
def test_reshape(inp, pruned_inp, enable_pruning):
    """Pass shape in forward."""
    op = Reshape(enable_pruning=True)
    shape = torch.Size((35, 2, 100))
    out = op(inp, shape)
    assert out.shape == shape

    # with the same input, the output shape should not change
    out = op(inp, shape)
    assert out.shape == shape

    # if input changes due to pruning, reshape should work
    # and output shape should change accordingly
    expected_shape = torch.Size((35, 2, 80))
    out = op(pruned_inp, shape)
    assert out.shape == expected_shape


@pytest.mark.parametrize("enable_pruning", [True, False])
def test_reshape_2(inp, pruned_inp, enable_pruning):
    """Pass shape in init."""
    shape = torch.Size((35, 2, 100))
    op = Reshape(enable_pruning=True, shape=shape)
    out = op(inp)
    assert out.shape == shape

    # input changes due to pruning, reshape should work
    expected_shape = torch.Size((35, 2, 80))
    out = op(pruned_inp)
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "shape,target",
    [((0, 3, 4), [3, 0, 4]), ((0, 3), [3, 0]), ((2, 0), [0, 2])],
)
def test_reshape_allowzero(shape, target):
    """A zero copies the input dimension by default, allowzero asks for it as
    the literal size instead."""
    x = np.zeros(shape, dtype=np.float32)
    model = make_single_node_model(
        "Reshape",
        {"x": x},
        21,
        initializers={"shape": np.array(target, dtype=np.int64)},
        allowzero=1,
    )
    assert_matches_oracle(model, {"x": x})


def test_reshape_zero_copies_the_input_dimension():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    model = make_single_node_model(
        "Reshape",
        {"x": x},
        21,
        initializers={"shape": np.array([0, 4], dtype=np.int64)},
    )
    assert_matches_oracle(model, {"x": x})
