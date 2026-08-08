import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_eye_like(x, dtype=None, k=None):
    attrs = {}
    if dtype is not None:
        attrs["dtype"] = dtype
    if k is not None:
        attrs["k"] = k
    node = helper.make_node("EyeLike", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "eyelike_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [
            helper.make_tensor_value_info(
                "y", dtype if dtype is not None else TensorProto.FLOAT, None
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_array_equal(y.numpy(), exp_y)


@pytest.mark.parametrize("k", [None, 0, 1, 2, -1, -2])
def test_eye_like_square(k):
    np.random.seed(0)
    check_eye_like(np.random.randn(4, 4).astype(np.float32), k=k)


@pytest.mark.parametrize("shape", [(3, 5), (5, 3)])
def test_eye_like_rectangular(shape):
    np.random.seed(0)
    check_eye_like(np.random.randn(*shape).astype(np.float32), k=1)


@pytest.mark.parametrize(
    "dtype",
    [TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT32, TensorProto.INT64],
)
def test_eye_like_dtype(dtype):
    np.random.seed(0)
    check_eye_like(np.random.randn(3, 3).astype(np.float32), dtype=dtype)
