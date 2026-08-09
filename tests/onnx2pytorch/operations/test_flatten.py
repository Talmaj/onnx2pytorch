import numpy as np
import pytest
import torch

from onnx2pytorch.operations import Flatten
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.fixture
def inp():
    return torch.rand(1, 3, 10, 10)


def test_flatten(inp):
    op = Flatten(1)
    out = op(inp)
    assert list(out.shape) == [1, 300]


def test_flatten_axis_two(inp):
    """ONNX Flatten is always 2D, unlike torch.flatten(start_dim=axis)."""
    op = Flatten(2)
    out = op(inp)
    assert list(out.shape) == [3, 100]


@pytest.mark.parametrize("opset_version", [1, 9, 11, 13, 21])
@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4])
def test_flatten_matches_onnxruntime(opset_version, axis):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    model = make_single_node_model("Flatten", {"x": x}, opset_version, axis=axis)
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("axis", [-1, -2, -4])
def test_flatten_negative_axis(axis):
    """Negative axes are legal from opset 11 on."""
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    model = make_single_node_model("Flatten", {"x": x}, 13, axis=axis)
    assert_matches_oracle(model, {"x": x})
