"""Graph level tests: a value has to reach every node that reads it, unchanged."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
    run_converted,
)


def value(name, shape=None, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def build(nodes, inputs, outputs, initializers=(), opset_version=14):
    graph = helper.make_graph(
        nodes, "test", inputs, outputs, initializer=list(initializers)
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )


@pytest.mark.parametrize("activation", ["Relu", "Elu", "LeakyRelu"])
def test_activation_does_not_consume_a_shared_input(activation):
    """These used to run in place, which rewrote the input for the other reader."""
    x = np.array([-3.0, 2.0], dtype=np.float32)
    model = build(
        [
            helper.make_node(activation, ["x"], ["activated"]),
            helper.make_node("Add", ["x", "one"], ["shifted"]),
        ],
        [value("x", [2])],
        [value("activated", [2]), value("shifted", [2])],
        [numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one")],
    )
    assert_matches_oracle(model, {"x": x})


def test_matmul_bias_is_only_folded_into_the_add_that_reads_it():
    """The next node used to be folded in as a bias whether it read the product
    or not, which dropped its real operands."""
    weight = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    bias = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    model = build(
        [
            helper.make_node("MatMul", ["x", "weight"], ["product"]),
            helper.make_node("Add", ["z", "bias"], ["shifted"]),
            helper.make_node("Mul", ["product", "shifted"], ["y"]),
        ],
        [value("x", [1, 1]), value("z", [3])],
        [value("y")],
        [
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    inputs = {
        "x": np.array([[2.0]], dtype=np.float32),
        "z": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    assert_matches_oracle(model, inputs)


def test_matmul_bias_is_not_folded_when_the_product_has_another_reader():
    weight = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    bias = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    model = build(
        [
            helper.make_node("MatMul", ["x", "weight"], ["product"]),
            helper.make_node("Add", ["product", "bias"], ["shifted"]),
            helper.make_node("Mul", ["product", "product"], ["squared"]),
        ],
        [value("x", [1, 1])],
        [value("shifted"), value("squared")],
        [
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    assert_matches_oracle(model, {"x": np.array([[2.0]], dtype=np.float32)})


def test_matmul_bias_is_folded():
    weight = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    bias = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    model = build(
        [
            helper.make_node("MatMul", ["x", "weight"], ["product"]),
            helper.make_node("Add", ["product", "bias"], ["y"]),
        ],
        [value("x", [1, 1])],
        [value("y")],
        [
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    assert_matches_oracle(model, {"x": np.array([[2.0]], dtype=np.float32)})


def test_intermediate_value_survives_as_a_graph_output():
    """Activations were dropped once their last reader had run, graph outputs
    among them."""
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    model = build(
        [
            helper.make_node("Abs", ["x"], ["absolute"]),
            helper.make_node("Relu", ["absolute"], ["y"]),
        ],
        [value("x", [3])],
        [value("y", [3]), value("absolute", [3])],
    )
    assert_matches_oracle(model, {"x": x})


def test_graph_input_survives_as_a_graph_output():
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    model = build(
        [helper.make_node("Abs", ["x"], ["absolute"])],
        [value("x", [3])],
        [value("absolute", [3]), value("x", [3])],
    )
    assert_matches_oracle(model, {"x": x})


def test_unknown_input_name_raises():
    """A name that is neither an activation nor an initializer used to silently
    resolve to the first graph input."""
    model = make_single_node_model("Abs", {"x": np.zeros(3, dtype=np.float32)}, 14)
    model.graph.node[0].input[0] = "nowhere"
    with pytest.raises(KeyError, match="nowhere"):
        run_converted(model, {"x": np.zeros(3, dtype=np.float32)})
