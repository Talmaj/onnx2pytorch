import numpy as np
import torch
import pytest

from onnx2pytorch.operations.squeeze import Squeeze
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.fixture
def inp():
    return torch.ones(1, 2, 1, 2)


@pytest.mark.parametrize(
    "dim, exp_shape",
    [
        (None, (2, 2)),
        (0, (2, 1, 2)),
        (2, (1, 2, 2)),
        (-2, (1, 2, 2)),
        (torch.tensor([0, 2]), (2, 2)),
    ],
)
def test_squeeze_v11(inp, dim, exp_shape):
    op = Squeeze(opset_version=11, dim=dim)
    assert tuple(op(inp).shape) == exp_shape


@pytest.mark.parametrize(
    "dim, exp_shape",
    [
        (None, (2, 2)),
        (0, (2, 1, 2)),
        (2, (1, 2, 2)),
        (-2, (1, 2, 2)),
        (torch.tensor([0, 2]), (2, 2)),
    ],
)
def test_squeeze_v13(inp, dim, exp_shape):
    op = Squeeze(opset_version=13)
    assert tuple(op(inp, dim).shape) == exp_shape


@pytest.mark.parametrize(
    "shape,axes",
    [
        ((1, 1, 3), [-3, 1]),
        ((1, 1, 3), [0, -2]),
        ((1, 3, 1, 4), [-4, -2]),
        ((1, 3, 1, 4), [-2, -4]),
        ((1, 1, 1), [-3, -2, -1]),
        ((1, 3, 1), [-1]),
    ],
)
def test_squeeze_negative_axes_count_against_the_input_rank(shape, axes):
    """Axes used to be applied against the shrinking input, so several negative
    ones ran off the end of it."""
    np.random.seed(0)
    x = np.random.randn(*shape).astype(np.float32)
    axes_tensor = np.array(axes, dtype=np.int64)
    model = make_single_node_model(
        "Squeeze", {"x": x}, 13, initializers={"axes": axes_tensor}
    )
    assert_matches_oracle(model, {"x": x})


def test_squeeze_empty_axes_tensor_squeezes_nothing():
    """An explicitly empty axes tensor, where the runtimes disagree.

    The spec only says what an *absent* axes means (squeeze every unit
    dimension), so an empty tensor is up for interpretation. onnxruntime's kernel
    treats it as absent, but its own shape inference for the same node keeps the
    input shape, and the onnx reference evaluator squeezes nothing. Naming no
    axis is taken to squeeze no axis, which is where those two agree.
    """
    x = torch.randn(1, 3, 1)
    y = Squeeze(opset_version=13)(x, torch.tensor([], dtype=torch.int64))
    assert y.shape == x.shape
