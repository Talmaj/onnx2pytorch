import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_regex_full_match(x, pattern):
    node = helper.make_node(
        "RegexFullMatch", inputs=["x"], outputs=["y"], pattern=pattern
    )
    graph = helper.make_graph(
        [node],
        "regexfullmatch_test",
        [helper.make_tensor_value_info("x", TensorProto.STRING, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.BOOL, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    with torch.no_grad():
        y = ConvertModel(model)(x)
    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_regex_full_match_basic():
    x = np.array(["www.google.com", "www.facebook.com", "www.bbc.co.uk"], dtype=object)
    check_regex_full_match(x, r"www\.[\w.]+\.\w+")


def test_regex_full_match_email_domain():
    x = np.array(
        [
            ["account@gmail.com", "account@hotmail.com"],
            ["not email", "account2@yahoo.com"],
        ],
        dtype=object,
    )
    check_regex_full_match(x, r"(\W|^)[\w.\-]{0,25}@(yahoo|gmail)\.com(\W|$)")


def test_regex_full_match_empty_match():
    x = np.array(["x", "y", "z"], dtype=object)
    check_regex_full_match(x, r"\d+")


def test_regex_full_match_partial_is_not_full():
    x = np.array(["abcd", "abc", "zabc"], dtype=object)
    check_regex_full_match(x, r"abc")


@pytest.mark.parametrize(
    "pattern", [r"[a-z]+", r"a*b", r"(foo|bar)", r"^abc$", r"a.c", r".*"]
)
def test_regex_full_match_patterns(pattern):
    x = np.array(["abc", "aab", "foo", "bar", "", "a c"], dtype=object)
    check_regex_full_match(x, pattern)


def test_regex_full_match_unicode():
    # RE2 restricts \w to ASCII, unlike Python's default unicode semantics
    x = np.array(["\u00e9t\u00e9", "ete", "\u4f60\u597d"], dtype=object)
    check_regex_full_match(x, r"\w+")


def test_regex_full_match_unicode_literal():
    x = np.array(["\u4f60\u597d", "abc"], dtype=object)
    check_regex_full_match(x, "\u4f60\u597d")


def test_regex_full_match_scalar():
    x = np.array("abc", dtype=object)
    check_regex_full_match(x, r"[a-c]+")
