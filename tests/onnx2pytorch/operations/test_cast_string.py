import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations import Cast


def build_cast_model(x, from_type, to_type, castlike=False):
    inputs = [helper.make_tensor_value_info("x", from_type, list(x.shape))]
    if castlike:
        node = helper.make_node("CastLike", inputs=["x", "target"], outputs=["y"])
        inputs.append(helper.make_tensor_value_info("target", to_type, [1]))
    else:
        node = helper.make_node("Cast", inputs=["x"], outputs=["y"], to=to_type)
    graph = helper.make_graph(
        [node],
        "cast_string_test",
        inputs,
        [helper.make_tensor_value_info("y", to_type, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


def check_cast(x, from_type, to_type, target=None):
    model = build_cast_model(x, from_type, to_type, castlike=target is not None)
    feed = {"x": x} if target is None else {"x": x, "target": target}

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(**feed)
    y = y.numpy() if torch.is_tensor(y) else y
    np.testing.assert_array_equal(y, exp_y)


def test_cast_int_to_string():
    x = np.array([1, -2, 30], dtype=np.int64)
    check_cast(x, TensorProto.INT64, TensorProto.STRING)


def test_cast_string_to_int():
    x = np.array(["1", "-2", "30"], dtype=object)
    check_cast(x, TensorProto.STRING, TensorProto.INT64)


def test_cast_string_to_float():
    x = np.array(["1.5", "-2", "NaN", "INF"], dtype=object)
    check_cast(x, TensorProto.STRING, TensorProto.FLOAT)


def test_cast_string_to_string():
    x = np.array(["a", "b"], dtype=object)
    check_cast(x, TensorProto.STRING, TensorProto.STRING)


def test_cast_like_int_to_string():
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    check_cast(
        x, TensorProto.INT32, TensorProto.STRING, target=np.array([""], dtype=object)
    )


def test_cast_like_string_to_int():
    x = np.array(["7", "8"], dtype=object)
    check_cast(
        x, TensorProto.STRING, TensorProto.INT64, target=np.zeros(1, dtype=np.int64)
    )


def test_cast_float_to_string_not_implemented():
    with pytest.raises(NotImplementedError):
        Cast("string")(torch.tensor([1.0, 2.5]))


def test_cast_into_string_operator():
    cast = helper.make_node("Cast", inputs=["x"], outputs=["s"], to=TensorProto.STRING)
    concat = helper.make_node("StringConcat", inputs=["s", "suffix"], outputs=["y"])
    graph = helper.make_graph(
        [cast, concat],
        "cast_stringconcat_test",
        [
            helper.make_tensor_value_info("x", TensorProto.INT64, [3]),
            helper.make_tensor_value_info("suffix", TensorProto.STRING, [3]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.STRING, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])

    feed = {
        "x": np.array([1, 2, 3], dtype=np.int64),
        "suffix": np.array(["a", "b", "c"], dtype=object),
    }
    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        y = ConvertModel(model)(**feed)
    np.testing.assert_array_equal(y, exp_y)
