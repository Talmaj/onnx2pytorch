import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


def check_hardmax(x, **attrs):
    node = helper.make_node("Hardmax", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "hardmax_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_hardmax():
    x = np.array(
        [[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2], [0, 1, 2, 3]], dtype=np.float32
    )
    check_hardmax(x)


def test_hardmax_ties():
    # Only the first occurrence of the maximum is set to 1
    x = np.array([[3, 3, 3, 1]], dtype=np.float32)
    check_hardmax(x)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_hardmax_axis(axis):
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    check_hardmax(x, axis=axis)


@pytest.mark.parametrize("axis", [None, 0, 1, 2, -1, -2])
@pytest.mark.parametrize("opset_version", [1, 11, 12, 13, 21])
def test_hardmax_across_opsets(axis, opset_version):
    """Before opset 13 the input is coerced to 2D around axis, default axis 1."""
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    attributes = {} if axis is None else {"axis": axis}
    model = make_single_node_model("Hardmax", {"x": x}, opset_version, **attributes)
    assert_matches_oracle(model, {"x": x})
