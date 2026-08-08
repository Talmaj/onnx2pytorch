import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_det(x):
    node = helper.make_node("Det", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "det_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_det_2d():
    check_det(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_det_batched():
    check_det(
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [2.0, 1.0]]], np.float32)
    )
    np.random.seed(0)
    check_det(np.random.randn(3, 4, 4).astype(np.float32))
