import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_shrink(x, **attrs):
    node = helper.make_node("Shrink", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "shrink_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "attrs",
    [{}, {"lambd": 1.5}, {"bias": 1.5, "lambd": 1.5}, {"bias": 0.5}],
)
def test_shrink(attrs):
    check_shrink(np.arange(-5, 6).astype(np.float32), **attrs)
    np.random.seed(0)
    check_shrink(np.random.uniform(-3, 3, (3, 4, 5)).astype(np.float32), **attrs)
