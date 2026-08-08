import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_argmax(x, **attrs):
    node = helper.make_node("ArgMax", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "argmax_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.INT64, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_array_equal(y.numpy(), exp_y)


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_argmax(keepdims, axis):
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    attrs = {"keepdims": keepdims}
    if axis is not None:
        attrs["axis"] = axis
    check_argmax(x, **attrs)


@pytest.mark.parametrize("select_last_index", [0, 1])
def test_argmax_ties(select_last_index):
    x = np.array([[2, 2], [3, 10]], dtype=np.float32)
    check_argmax(x, axis=1, keepdims=1, select_last_index=select_last_index)
    check_argmax(x, axis=0, keepdims=0, select_last_index=select_last_index)
