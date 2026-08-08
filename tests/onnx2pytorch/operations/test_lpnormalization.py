import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_lp_normalization(x, axis, p):
    node = helper.make_node(
        "LpNormalization", inputs=["x"], outputs=["y"], axis=axis, p=p
    )
    graph = helper.make_graph(
        [node],
        "lpnormalization_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_lp_normalization(axis, p):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    check_lp_normalization(x, axis, p)


def test_lp_normalization_default_attributes():
    np.random.seed(0)
    x = np.random.randn(3, 5).astype(np.float32)
    node = helper.make_node("LpNormalization", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "lpnormalization_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)
