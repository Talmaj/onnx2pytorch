import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_nonzero(x, elem_type=TensorProto.FLOAT):
    node = helper.make_node("NonZero", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "nonzero_test",
        [helper.make_tensor_value_info("x", elem_type, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.INT64, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_array_equal(y.numpy(), exp_y)


@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
def test_nonzero(shape):
    np.random.seed(0)
    x = (np.random.rand(*shape) > 0.5).astype(np.float32)
    check_nonzero(x)


def test_nonzero_bool():
    np.random.seed(0)
    x = np.random.rand(3, 4) > 0.5
    check_nonzero(x, TensorProto.BOOL)


def test_nonzero_all_zeros():
    check_nonzero(np.zeros((2, 3), dtype=np.float32))
