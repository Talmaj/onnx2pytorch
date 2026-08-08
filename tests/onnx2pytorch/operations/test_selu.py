import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.selu import Selu


def check_selu(x, **attrs):
    node = helper.make_node("Selu", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "selu_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 6)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "attrs", [{}, {"alpha": 2.0, "gamma": 3.0}, {"alpha": 0.5}, {"gamma": 0.5}]
)
def test_selu(attrs):
    check_selu(np.array([-1.0, 0.0, 1.0], dtype=np.float32), **attrs)
    np.random.seed(0)
    check_selu(np.random.randn(3, 4, 5).astype(np.float32), **attrs)


def test_selu_defaults():
    op = Selu()
    x = torch.tensor([-1.0, 0.0, 1.0])
    exp_y = torch.nn.functional.selu(x)
    torch.testing.assert_close(op(x), exp_y, rtol=1e-6, atol=1e-6)
