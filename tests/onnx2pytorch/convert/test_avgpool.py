import numpy as np
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


@pytest.fixture
def x():
    np.random.seed(0)
    return np.random.randn(2, 3, 5, 5).astype(np.float32)


@pytest.mark.parametrize("opset_version", [7, 10, 11, 19, 22])
def test_average_pool_excludes_pads(x, opset_version):
    """Since opset 7 ONNX leaves the pads out of the average, unlike torch."""
    model = make_single_node_model(
        "AveragePool", {"x": x}, opset_version, kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("count_include_pad", [0, 1])
def test_average_pool_count_include_pad(x, count_include_pad):
    model = make_single_node_model(
        "AveragePool",
        {"x": x},
        19,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        count_include_pad=count_include_pad,
    )
    assert_matches_oracle(model, {"x": x})


def test_average_pool_includes_pads_before_opset_7(x):
    model = make_single_node_model(
        "AveragePool", {"x": x}, 1, kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("op_type", ["AveragePool", "MaxPool"])
def test_pool_stride_defaults_to_one(x, op_type):
    """ONNX strides default to 1 while torch's pooling defaults to kernel_size."""
    model = make_single_node_model(op_type, {"x": x}, 11, kernel_shape=[3, 3])
    assert_matches_oracle(model, {"x": x})


def test_average_pool_1d(x):
    data = x[:, :, 0, :]
    model = make_single_node_model(
        "AveragePool", {"x": data}, 11, kernel_shape=[3], pads=[1, 1]
    )
    assert_matches_oracle(model, {"x": data})
