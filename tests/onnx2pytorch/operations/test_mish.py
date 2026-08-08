import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_mish(x):
    node = helper.make_node("Mish", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "mish_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


def test_mish():
    check_mish(np.linspace(-10, 10, 21).astype(np.float32))
    np.random.seed(0)
    check_mish(np.random.randn(3, 4, 5).astype(np.float32))
