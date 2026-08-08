import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_bitwise_not(x, tensor_type):
    node = helper.make_node("BitwiseNot", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "bitwisenot_test",
        [helper.make_tensor_value_info("x", tensor_type, list(x.shape))],
        [helper.make_tensor_value_info("y", tensor_type, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_array_equal(y.numpy(), exp_y)


@pytest.mark.parametrize(
    "np_dtype, tensor_type",
    [
        (np.int32, TensorProto.INT32),
        (np.int16, TensorProto.INT16),
        (np.uint8, TensorProto.UINT8),
    ],
)
def test_bitwise_not(np_dtype, tensor_type):
    np.random.seed(0)
    x = np.random.randint(0, 100, (3, 4, 5)).astype(np_dtype)
    check_bitwise_not(x, tensor_type)
