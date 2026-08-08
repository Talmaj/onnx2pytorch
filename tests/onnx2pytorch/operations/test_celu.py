import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_celu(x, **attrs):
    node = helper.make_node("Celu", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "celu_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("attrs", [{}, {"alpha": 2.0}, {"alpha": 0.5}])
def test_celu(attrs):
    x = np.array([[[[0.8, 1.2], [-0.5, -2.0]]]], dtype=np.float32)
    check_celu(x, **attrs)
    np.random.seed(0)
    check_celu(np.random.randn(3, 4, 5).astype(np.float32), **attrs)
