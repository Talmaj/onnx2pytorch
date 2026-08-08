import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_optional_has_element(nodes, x):
    graph = helper.make_graph(
        nodes,
        "optionalhaselement_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("has", TensorProto.BOOL, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    exp_has = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        has = o2p_model(torch.from_numpy(x))

    np.testing.assert_equal(has.numpy(), exp_has)


TENSOR_TYPE_PROTO = helper.make_tensor_type_proto(TensorProto.FLOAT, [3])
X = np.array([1.0, 2.0, 3.0], dtype=np.float32)


def test_optional_has_element_non_empty():
    nodes = [
        helper.make_node("Optional", ["x"], ["opt"]),
        helper.make_node("OptionalHasElement", ["opt"], ["has"]),
    ]
    check_optional_has_element(nodes, X)


def test_optional_has_element_empty():
    nodes = [
        helper.make_node("Optional", [], ["opt"], type=TENSOR_TYPE_PROTO),
        helper.make_node("OptionalHasElement", ["opt"], ["has"]),
    ]
    check_optional_has_element(nodes, X)


def test_optional_has_element_omitted_input():
    nodes = [helper.make_node("OptionalHasElement", [], ["has"])]
    check_optional_has_element(nodes, X)


def test_optional_has_element_tensor_input():
    nodes = [helper.make_node("OptionalHasElement", ["x"], ["has"])]
    check_optional_has_element(nodes, X)


def test_optional_has_element_sequence_optional():
    nodes = [
        helper.make_node("SequenceConstruct", ["x", "x"], ["seq"]),
        helper.make_node("Optional", ["seq"], ["opt"]),
        helper.make_node("OptionalHasElement", ["opt"], ["has"]),
    ]
    check_optional_has_element(nodes, X)


def test_optional_has_element_empty_sequence_counts_as_element():
    nodes = [
        helper.make_node("SequenceEmpty", [], ["seq"], dtype=TensorProto.FLOAT),
        helper.make_node("Optional", ["seq"], ["opt"]),
        helper.make_node("OptionalHasElement", ["opt"], ["has"]),
    ]
    check_optional_has_element(nodes, X)
