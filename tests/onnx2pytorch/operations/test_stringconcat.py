import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def build_model(x, y, y_as_initializer=False):
    node = helper.make_node("StringConcat", inputs=["x", "y"], outputs=["z"])
    inputs = [helper.make_tensor_value_info("x", TensorProto.STRING, list(x.shape))]
    initializers = []
    if y_as_initializer:
        initializers.append(
            helper.make_tensor(
                "y",
                TensorProto.STRING,
                list(y.shape),
                [s.encode() for s in y.reshape(-1)],
            )
        )
    else:
        inputs.append(
            helper.make_tensor_value_info("y", TensorProto.STRING, list(y.shape))
        )
    graph = helper.make_graph(
        [node],
        "stringconcat_test",
        inputs,
        [helper.make_tensor_value_info("z", TensorProto.STRING, None)],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def check_string_concat(x, y, y_as_initializer=False):
    model = build_model(x, y, y_as_initializer)

    feed = {"x": x} if y_as_initializer else {"x": x, "y": y}
    exp_z = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        z = ConvertModel(model)(**feed)
    np.testing.assert_array_equal(z, exp_z)


def test_string_concat():
    x = np.array(["abc", "def"], dtype=object)
    y = np.array([".com", ".net"], dtype=object)
    check_string_concat(x, y)


def test_string_concat_empty_string():
    x = np.array(["cat", "dog", ""], dtype=object)
    y = np.array(["s", "", "fish"], dtype=object)
    check_string_concat(x, y)


def test_string_concat_broadcasting():
    x = np.array(["abc", "def"], dtype=object)
    y = np.array(["abc"], dtype=object)
    check_string_concat(x, y)


def test_string_concat_2d_broadcasting():
    x = np.array([["a", "b"]], dtype=object)
    y = np.array([["c"], ["d"]], dtype=object)
    check_string_concat(x, y)


def test_string_concat_scalar():
    x = np.array("abc", dtype=object)
    y = np.array("def", dtype=object)
    check_string_concat(x, y)


def test_string_concat_unicode():
    x = np.array(["\u4f60\u597d", "\u00e9t\u00e9"], dtype=object)
    y = np.array(["\u4e16\u754c", "!"], dtype=object)
    check_string_concat(x, y)


def test_string_concat_initializer():
    x = np.array(["abc", "def"], dtype=object)
    y = np.array([".com", ".net"], dtype=object)
    check_string_concat(x, y, y_as_initializer=True)
