import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_isinf(x, **attrs):
    node = helper.make_node("IsInf", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "isinf_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.BOOL, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_array_equal(y.numpy(), exp_y)


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"detect_negative": 0},
        {"detect_positive": 0},
        {"detect_negative": 0, "detect_positive": 0},
    ],
)
def test_isinf(attrs):
    x = np.array([-1.2, np.nan, np.inf, 2.8, -np.inf, np.inf], dtype=np.float32)
    check_isinf(x, **attrs)
