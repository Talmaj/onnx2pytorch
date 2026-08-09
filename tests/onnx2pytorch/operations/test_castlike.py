import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_cast_like(x, target):
    node = helper.make_node("CastLike", inputs=["x", "target"], outputs=["y"])
    target_type = helper.np_dtype_to_tensor_dtype(target.dtype)
    graph = helper.make_graph(
        [node],
        "castlike_test",
        [
            helper.make_tensor_value_info(
                "x", helper.np_dtype_to_tensor_dtype(x.dtype), list(x.shape)
            ),
            helper.make_tensor_value_info("target", target_type, list(target.shape)),
        ],
        [helper.make_tensor_value_info("y", target_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])

    feed = {"x": x, "target": target}
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    assert y.numpy().dtype == exp_y.dtype
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_cast_like_float_to_int():
    x = np.array([[1.7, -2.3], [0.0, 5.9]], dtype=np.float32)
    check_cast_like(x, np.zeros((1,), dtype=np.int64))


def test_cast_like_int_to_float():
    x = np.array([[1, -2], [0, 5]], dtype=np.int64)
    check_cast_like(x, np.zeros((1,), dtype=np.float32))


def test_cast_like_float_to_double():
    np.random.seed(0)
    x = np.random.randn(3, 4).astype(np.float32)
    check_cast_like(x, np.zeros((2, 2), dtype=np.float64))


def test_cast_like_double_to_float16():
    np.random.seed(1)
    x = np.random.randn(2, 3).astype(np.float64)
    check_cast_like(x, np.zeros((1,), dtype=np.float16))


def test_cast_like_to_bool():
    x = np.array([0.0, 1.0, -3.0, 0.0], dtype=np.float32)
    check_cast_like(x, np.zeros((1,), dtype=bool))


def test_cast_like_bool_to_int():
    x = np.array([True, False, True], dtype=bool)
    check_cast_like(x, np.zeros((1,), dtype=np.int32))


def test_cast_like_same_type_is_identity():
    np.random.seed(2)
    x = np.random.randn(2, 3).astype(np.float32)
    check_cast_like(x, np.zeros((5,), dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32])
def test_cast_like_integer_types(dtype):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    check_cast_like(x, np.zeros((1,), dtype=dtype))
