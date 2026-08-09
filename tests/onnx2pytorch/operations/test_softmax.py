import numpy as np
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)

SHAPES = [(2, 3, 4), (5, 10), (3, 2, 2, 2)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 21])
def test_softmax_default_axis(shape, opset_version):
    """The default axis is 1 before opset 13 and -1 from 13 on."""
    np.random.seed(0)
    x = np.random.randn(*shape).astype(np.float32)
    model = make_single_node_model("Softmax", {"x": x}, opset_version)
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2])
@pytest.mark.parametrize("opset_version", [11, 12, 13, 21])
def test_softmax_axis(axis, opset_version):
    """Before opset 13 the input is coerced to 2D around axis."""
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    model = make_single_node_model("Softmax", {"x": x}, opset_version, axis=axis)
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_softmax_opset_1(axis):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    model = make_single_node_model("Softmax", {"x": x}, 1, axis=axis)
    assert_matches_oracle(model, {"x": x})
