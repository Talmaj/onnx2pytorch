import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_string_split(x, **attrs):
    node = helper.make_node(
        "StringSplit", inputs=["x"], outputs=["substrings", "length"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "stringsplit_test",
        [helper.make_tensor_value_info("x", TensorProto.STRING, list(x.shape))],
        [
            helper.make_tensor_value_info("substrings", TensorProto.STRING, None),
            helper.make_tensor_value_info("length", TensorProto.INT64, None),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])

    exp_substrings, exp_length = ort.InferenceSession(model.SerializeToString()).run(
        None, {"x": x}
    )
    with torch.no_grad():
        substrings, length = ConvertModel(model)(x)
    assert substrings.shape == exp_substrings.shape
    np.testing.assert_array_equal(substrings, exp_substrings)
    np.testing.assert_array_equal(length.numpy(), exp_length)


def test_string_split_basic():
    x = np.array(["hello world", "def.net", ""], dtype=object)
    check_string_split(x, delimiter=".")


def test_string_split_default_whitespace():
    x = np.array(["hello world", "  a  b  ", "def", ""], dtype=object)
    check_string_split(x)


def test_string_split_empty_delimiter_is_whitespace():
    x = np.array(["hello world", " a  b ", "def"], dtype=object)
    check_string_split(x, delimiter="")


def test_string_split_non_space_whitespace():
    from onnx2pytorch.operations import StringSplit

    # The spec splits on any consecutive whitespace, onnxruntime only on spaces
    substrings, length = StringSplit()(np.array(["a\tb", "c"], dtype=object))
    np.testing.assert_array_equal(substrings, np.array([["a", "b"], ["c", ""]]))
    np.testing.assert_array_equal(length.numpy(), np.array([2, 1]))


def test_string_split_maxsplit():
    x = np.array(["a*b*c", "d", "*e*"], dtype=object)
    check_string_split(x, delimiter="*", maxsplit=1)


@pytest.mark.parametrize("maxsplit", [0, 1, 2, 5])
def test_string_split_maxsplit_values(maxsplit):
    x = np.array(["a.b.c.d", "e.f", "g"], dtype=object)
    check_string_split(x, delimiter=".", maxsplit=maxsplit)


def test_string_split_multi_character_delimiter():
    x = np.array(["a--b--c", "d--e", "f"], dtype=object)
    check_string_split(x, delimiter="--")


def test_string_split_no_match():
    x = np.array(["abc", "def"], dtype=object)
    check_string_split(x, delimiter=";")


def test_string_split_2d():
    x = np.array([["a.b", "c"], ["d.e.f", ""]], dtype=object)
    check_string_split(x, delimiter=".")


def test_string_split_scalar():
    x = np.array("a,b,c", dtype=object)
    check_string_split(x, delimiter=",")


def test_string_split_all_empty_strings():
    x = np.array(["", ""], dtype=object)
    check_string_split(x, delimiter=".")


def test_string_split_empty_tensor():
    x = np.array([], dtype=object)
    check_string_split(x, delimiter=".")


def test_string_split_consecutive_delimiters():
    x = np.array(["a..b", "..", "a.b"], dtype=object)
    check_string_split(x, delimiter=".")
