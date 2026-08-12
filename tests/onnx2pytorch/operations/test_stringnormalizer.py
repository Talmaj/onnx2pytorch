import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations import StringNormalizer


def check_string_normalizer(x, **attrs):
    node = helper.make_node("StringNormalizer", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "stringnormalizer_test",
        [helper.make_tensor_value_info("x", TensorProto.STRING, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.STRING, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    with torch.no_grad():
        y = ConvertModel(model)(x)
    assert y.shape == exp_y.shape
    np.testing.assert_array_equal(y, exp_y)


@pytest.mark.parametrize("case_change_action", ["NONE", "LOWER", "UPPER"])
def test_string_normalizer_case_change_action(case_change_action):
    x = np.array(["monday", "tuesday", "wednesday", "thursday"], dtype=object)
    check_string_normalizer(
        x,
        case_change_action=case_change_action,
        is_case_sensitive=1,
        stopwords=["tuesday"],
    )


def test_string_normalizer_no_attributes():
    x = np.array(["Monday", "tuesday"], dtype=object)
    check_string_normalizer(x)


def test_string_normalizer_case_sensitive():
    x = np.array(["monday", "tuesday", "Monday", "Tuesday"], dtype=object)
    check_string_normalizer(
        x, case_change_action="NONE", is_case_sensitive=1, stopwords=["monday"]
    )


def test_string_normalizer_case_insensitive():
    x = np.array(["monday", "tuesday", "Monday", "Tuesday"], dtype=object)
    check_string_normalizer(
        x, case_change_action="NONE", is_case_sensitive=0, stopwords=["monday"]
    )


def test_string_normalizer_uppercase_stopwords():
    x = np.array(["monday", "tuesday"], dtype=object)
    check_string_normalizer(
        x, case_change_action="UPPER", is_case_sensitive=1, stopwords=["MONDAY"]
    )


def test_string_normalizer_multiple_stopwords():
    x = np.array(["monday", "tuesday", "wednesday", "thursday"], dtype=object)
    check_string_normalizer(
        x,
        case_change_action="LOWER",
        is_case_sensitive=0,
        stopwords=["Monday", "THURSDAY"],
    )


def test_string_normalizer_multiword_elements():
    x = np.array(["a b", "a", "c"], dtype=object)
    check_string_normalizer(x, case_change_action="LOWER", stopwords=["a"])


def test_string_normalizer_all_stopwords_1d():
    x = np.array(["monday", "tuesday"], dtype=object)
    check_string_normalizer(
        x,
        case_change_action="UPPER",
        is_case_sensitive=0,
        stopwords=["monday", "tuesday"],
    )


def test_string_normalizer_all_stopwords_2d():
    x = np.array([["monday", "tuesday"]], dtype=object)
    check_string_normalizer(
        x,
        case_change_action="UPPER",
        is_case_sensitive=0,
        stopwords=["monday", "tuesday"],
    )


@pytest.mark.parametrize("case_change_action", ["NONE", "LOWER", "UPPER"])
def test_string_normalizer_2d(case_change_action):
    x = np.array([["monday", "tuesday", "wednesday", "thursday"]], dtype=object)
    check_string_normalizer(
        x,
        case_change_action=case_change_action,
        is_case_sensitive=0,
        stopwords=["wednesday"],
    )


def test_string_normalizer_locale_is_ignored():
    # ORT constructs a C++ locale from this attribute and fails on many CI
    # images (missing en_US), so assert directly that we accept and ignore it.
    x = np.array(["Monday", "tuesday"], dtype=object)
    expected = np.array(["MONDAY", "TUESDAY"], dtype=object)
    np.testing.assert_array_equal(
        StringNormalizer(case_change_action="UPPER", locale="en_US")(x), expected
    )
    node = helper.make_node(
        "StringNormalizer",
        inputs=["x"],
        outputs=["y"],
        case_change_action="UPPER",
        locale="en_US",
    )
    graph = helper.make_graph(
        [node],
        "stringnormalizer_locale_test",
        [helper.make_tensor_value_info("x", TensorProto.STRING, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.STRING, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    with torch.no_grad():
        y = ConvertModel(model)(x)
    np.testing.assert_array_equal(y, expected)


def test_string_normalizer_invalid_case_change_action():
    with pytest.raises(NotImplementedError):
        StringNormalizer(case_change_action="TITLE")


def test_string_normalizer_unsupported_shape():
    x = np.array([["a"], ["b"]], dtype=object)
    with pytest.raises(ValueError):
        StringNormalizer()(x)
