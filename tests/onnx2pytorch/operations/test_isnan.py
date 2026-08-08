import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_isnan(x):
    node = helper.make_node("IsNaN", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "isnan_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.BOOL, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_isnan():
    check_isnan(np.array([3.0, np.nan, 4.0, np.nan], dtype=np.float32))
    check_isnan(np.array([[-1.2, np.nan], [np.inf, -np.inf]], dtype=np.float32))
