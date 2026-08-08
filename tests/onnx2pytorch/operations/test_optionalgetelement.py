import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_optional_get_element(nodes, x):
    graph = helper.make_graph(
        nodes,
        "optionalgetelement_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_optional_get_element_tensor_optional():
    nodes = [
        helper.make_node("Optional", ["x"], ["opt"]),
        helper.make_node("OptionalGetElement", ["opt"], ["y"]),
    ]
    check_optional_get_element(nodes, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_optional_get_element_tensor_input():
    nodes = [helper.make_node("OptionalGetElement", ["x"], ["y"])]
    check_optional_get_element(nodes, np.random.randn(2, 3).astype(np.float32))


def test_optional_get_element_sequence_optional():
    nodes = [
        helper.make_node("SequenceConstruct", ["x", "x"], ["seq"]),
        helper.make_node("Optional", ["seq"], ["opt"]),
        helper.make_node("OptionalGetElement", ["opt"], ["seq_out"]),
        helper.make_node("ConcatFromSequence", ["seq_out"], ["y"], axis=0),
    ]
    check_optional_get_element(nodes, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_optional_get_element_followed_by_computation():
    nodes = [
        helper.make_node("Optional", ["x"], ["opt"]),
        helper.make_node("OptionalGetElement", ["opt"], ["elem"]),
        helper.make_node("Add", ["elem", "elem"], ["y"]),
    ]
    check_optional_get_element(nodes, np.random.randn(3, 2).astype(np.float32))


def test_optional_get_element_empty_optional_raises():
    from onnx2pytorch.operations import OptionalGetElement

    with pytest.raises(ValueError, match="empty optional"):
        OptionalGetElement()(None)
