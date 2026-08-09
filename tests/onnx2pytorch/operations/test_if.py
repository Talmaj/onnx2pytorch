import numpy as np
import onnx
import pytest
import torch
from onnx import helper, numpy_helper

from onnx2pytorch.operations.if_op import If
from tests.onnx2pytorch.differential import assert_matches_oracle


def test_if_basic_true():
    """Test If operator with condition=True."""
    # Create simple then and else branches
    then_const = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    else_const = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    then_out = helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    then_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["then_out"],
        value=numpy_helper.from_array(then_const),
    )

    else_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["else_out"],
        value=numpy_helper.from_array(else_const),
    )

    then_body = helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = helper.make_graph([else_const_node], "else_body", [], [else_out])

    # Create If operation
    op = If(
        opset_version=11,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    # Test with condition = True
    cond = torch.tensor(True)
    outputs = op((), {}, cond)

    assert len(outputs) == 1
    assert torch.allclose(outputs[0], torch.tensor(then_const))


def test_if_basic_false():
    """Test If operator with condition=False."""
    # Create simple then and else branches
    then_const = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    else_const = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    then_out = helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    then_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["then_out"],
        value=numpy_helper.from_array(then_const),
    )

    else_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["else_out"],
        value=numpy_helper.from_array(else_const),
    )

    then_body = helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = helper.make_graph([else_const_node], "else_body", [], [else_out])

    # Create If operation
    op = If(
        opset_version=11,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    # Test with condition = False
    cond = torch.tensor(False)
    outputs = op((), {}, cond)

    assert len(outputs) == 1
    assert torch.allclose(outputs[0], torch.tensor(else_const))


def test_if_with_numpy_bool():
    """Test If operator with numpy bool input."""
    then_const = np.array([1, 2, 3]).astype(np.float32)
    else_const = np.array([4, 5, 6]).astype(np.float32)

    then_out = helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [3])
    else_out = helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [3])

    then_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["then_out"],
        value=numpy_helper.from_array(then_const),
    )

    else_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["else_out"],
        value=numpy_helper.from_array(else_const),
    )

    then_body = helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = helper.make_graph([else_const_node], "else_body", [], [else_out])

    op = If(
        opset_version=11,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    # Test with numpy bool
    cond = np.array(1).astype(bool)
    outputs = op((), {}, cond)

    assert len(outputs) == 1
    assert torch.allclose(outputs[0], torch.tensor(then_const))


def test_if_official_onnx():
    """
    Official ONNX test case for If.
    Returns constant tensor x if cond is True, otherwise return constant tensor y.
    From: https://github.com/onnx/onnx/blob/main/docs/Operators.md#If
    """
    then_out = helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    then_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["then_out"],
        value=numpy_helper.from_array(x),
    )

    else_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["else_out"],
        value=numpy_helper.from_array(y),
    )

    then_body = helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = helper.make_graph([else_const_node], "else_body", [], [else_out])

    op = If(
        opset_version=11,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    cond = np.array(1).astype(bool)
    expected = x if cond else y

    outputs = op((), {}, cond)

    assert len(outputs) == 1
    assert torch.allclose(outputs[0], torch.tensor(expected))


def test_if_with_computation():
    """Test If with actual computation in branches (not just constants)."""
    # Then branch: add two constants
    then_out = helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [3])

    a = np.array([1, 2, 3]).astype(np.float32)
    b = np.array([4, 5, 6]).astype(np.float32)

    then_const_a = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["a"],
        value=numpy_helper.from_array(a),
    )

    then_const_b = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["b"],
        value=numpy_helper.from_array(b),
    )

    then_add = helper.make_node(
        "Add",
        inputs=["a", "b"],
        outputs=["then_out"],
    )

    then_body = helper.make_graph(
        [then_const_a, then_const_b, then_add], "then_body", [], [then_out]
    )

    # Else branch: multiply two constants
    else_out = helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [3])

    c = np.array([2, 3, 4]).astype(np.float32)
    d = np.array([10, 10, 10]).astype(np.float32)

    else_const_c = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c"],
        value=numpy_helper.from_array(c),
    )

    else_const_d = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["d"],
        value=numpy_helper.from_array(d),
    )

    else_mul = helper.make_node(
        "Mul",
        inputs=["c", "d"],
        outputs=["else_out"],
    )

    else_body = helper.make_graph(
        [else_const_c, else_const_d, else_mul], "else_body", [], [else_out]
    )

    op = If(
        opset_version=11,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    # Test then branch
    cond = torch.tensor(True)
    outputs = op((), {}, cond)
    expected = a + b
    assert torch.allclose(outputs[0], torch.tensor(expected))

    # Test else branch
    cond = torch.tensor(False)
    outputs = op((), {}, cond)
    expected = c * d
    assert torch.allclose(outputs[0], torch.tensor(expected))


def test_if_multiple_outputs():
    """Test If with multiple outputs from each branch."""
    # Then branch: returns two tensors
    then_out1 = helper.make_tensor_value_info("then_out1", onnx.TensorProto.FLOAT, [2])
    then_out2 = helper.make_tensor_value_info("then_out2", onnx.TensorProto.FLOAT, [2])

    x1 = np.array([1, 2]).astype(np.float32)
    x2 = np.array([3, 4]).astype(np.float32)

    then_const1 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["then_out1"],
        value=numpy_helper.from_array(x1),
    )

    then_const2 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["then_out2"],
        value=numpy_helper.from_array(x2),
    )

    then_body = helper.make_graph(
        [then_const1, then_const2], "then_body", [], [then_out1, then_out2]
    )

    # Else branch: returns two tensors
    else_out1 = helper.make_tensor_value_info("else_out1", onnx.TensorProto.FLOAT, [2])
    else_out2 = helper.make_tensor_value_info("else_out2", onnx.TensorProto.FLOAT, [2])

    y1 = np.array([5, 6]).astype(np.float32)
    y2 = np.array([7, 8]).astype(np.float32)

    else_const1 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["else_out1"],
        value=numpy_helper.from_array(y1),
    )

    else_const2 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["else_out2"],
        value=numpy_helper.from_array(y2),
    )

    else_body = helper.make_graph(
        [else_const1, else_const2], "else_body", [], [else_out1, else_out2]
    )

    op = If(
        opset_version=11,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    # Test then branch
    cond = torch.tensor(True)
    outputs = op((), {}, cond)
    assert len(outputs) == 2
    assert torch.allclose(outputs[0], torch.tensor(x1))
    assert torch.allclose(outputs[1], torch.tensor(x2))

    # Test else branch
    cond = torch.tensor(False)
    outputs = op((), {}, cond)
    assert len(outputs) == 2
    assert torch.allclose(outputs[0], torch.tensor(y1))
    assert torch.allclose(outputs[1], torch.tensor(y2))


def test_if_different_shapes():
    """Test If where then and else branches return different shapes."""
    # Then branch: returns [2] shape
    then_out = helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [2])

    x = np.array([1, 2]).astype(np.float32)

    then_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["then_out"],
        value=numpy_helper.from_array(x),
    )

    then_body = helper.make_graph([then_const], "then_body", [], [then_out])

    # Else branch: returns [3] shape
    else_out = helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [3])

    y = np.array([3, 4, 5]).astype(np.float32)

    else_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["else_out"],
        value=numpy_helper.from_array(y),
    )

    else_body = helper.make_graph([else_const], "else_body", [], [else_out])

    op = If(
        opset_version=11,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    # Test then branch
    cond = torch.tensor(True)
    outputs = op((), {}, cond)
    assert outputs[0].shape == (2,)
    assert torch.allclose(outputs[0], torch.tensor(x))

    # Test else branch
    cond = torch.tensor(False)
    outputs = op((), {}, cond)
    assert outputs[0].shape == (3,)
    assert torch.allclose(outputs[0], torch.tensor(y))


def test_if_seq_official_onnx():
    """
    Official ONNX test case for If with sequences.
    Returns constant sequence x if cond is True, otherwise return constant sequence y.
    From: https://github.com/onnx/onnx/blob/main/docs/Operators.md#If
    """
    then_out = helper.make_tensor_sequence_value_info(
        "then_out", onnx.TensorProto.FLOAT, shape=[5]
    )
    else_out = helper.make_tensor_sequence_value_info(
        "else_out", onnx.TensorProto.FLOAT, shape=[5]
    )

    x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
    y = [np.array([5, 4, 3, 2, 1]).astype(np.float32)]

    then_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["x"],
        value=numpy_helper.from_array(x[0]),
    )

    then_seq_node = helper.make_node(
        "SequenceConstruct", inputs=["x"], outputs=["then_out"]
    )

    else_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["y"],
        value=numpy_helper.from_array(y[0]),
    )

    else_seq_node = helper.make_node(
        "SequenceConstruct", inputs=["y"], outputs=["else_out"]
    )

    then_body = helper.make_graph(
        [then_const_node, then_seq_node], "then_body", [], [then_out]
    )

    else_body = helper.make_graph(
        [else_const_node, else_seq_node], "else_body", [], [else_out]
    )

    op = If(
        opset_version=13,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    cond = np.array(1).astype(bool)
    expected = x if cond else y

    outputs = op((), {}, cond)

    assert len(outputs) == 1
    # Output should be a sequence (list) containing one tensor
    assert isinstance(outputs[0], (list, tuple))
    assert len(outputs[0]) == 1
    assert torch.allclose(outputs[0][0], torch.tensor(expected[0]))


def test_if_opt_official_onnx():
    """
    Official ONNX test case for If with optional types.
    Return an empty optional sequence of tensor if True,
    return an optional sequence with value x otherwise.
    From: https://github.com/onnx/onnx/blob/main/docs/Operators.md#If
    """
    ten_in_tp = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=[5])
    seq_in_tp = helper.make_sequence_type_proto(ten_in_tp)

    then_out_tensor_tp = helper.make_tensor_type_proto(
        onnx.TensorProto.FLOAT, shape=[5]
    )
    then_out_seq_tp = helper.make_sequence_type_proto(then_out_tensor_tp)
    then_out_opt_tp = helper.make_optional_type_proto(then_out_seq_tp)
    then_out = helper.make_value_info("optional_empty", then_out_opt_tp)

    else_out_tensor_tp = helper.make_tensor_type_proto(
        onnx.TensorProto.FLOAT, shape=[5]
    )
    else_out_seq_tp = helper.make_sequence_type_proto(else_out_tensor_tp)
    else_out_opt_tp = helper.make_optional_type_proto(else_out_seq_tp)
    else_out = helper.make_value_info("else_opt", else_out_opt_tp)

    x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
    cond = np.array(0).astype(bool)

    opt_empty_in = helper.make_node(
        "Optional", inputs=[], outputs=["optional_empty"], type=seq_in_tp
    )
    then_body = helper.make_graph([opt_empty_in], "then_body", [], [then_out])

    else_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["x"],
        value=numpy_helper.from_array(x[0]),
    )

    else_seq_node = helper.make_node(
        "SequenceConstruct", inputs=["x"], outputs=["else_seq"]
    )

    else_optional_seq_node = helper.make_node(
        "Optional", inputs=["else_seq"], outputs=["else_opt"]
    )

    else_body = helper.make_graph(
        [else_const_node, else_seq_node, else_optional_seq_node],
        "else_body",
        [],
        [else_out],
    )

    op = If(
        opset_version=16,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    # This test requires Optional and SequenceConstruct operators to be implemented
    outputs = op((), {}, cond)

    # Expected: optional sequence with value x
    assert len(outputs) == 1


def test_if_opset_version_24():
    """Test If operator with opset version 24 (latest)."""
    then_out = helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [3])
    else_out = helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [3])

    x = np.array([10, 20, 30]).astype(np.float32)
    y = np.array([40, 50, 60]).astype(np.float32)

    then_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["then_out"],
        value=numpy_helper.from_array(x),
    )

    else_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["else_out"],
        value=numpy_helper.from_array(y),
    )

    then_body = helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = helper.make_graph([else_const_node], "else_body", [], [else_out])

    # Test with opset version 24
    op = If(
        opset_version=24,
        batch_dim=0,
        then_branch=then_body,
        else_branch=else_body,
    )

    cond = torch.tensor(True)
    outputs = op((), {}, cond)

    assert len(outputs) == 1
    assert torch.allclose(outputs[0], torch.tensor(x))


def make_if_model(opset_version, num_outputs=1):
    """Model whose If branches read x from the enclosing scope."""
    then_nodes = [helper.make_node("Mul", ["x", "two"], ["then_out"])]
    else_nodes = [helper.make_node("Add", ["x", "ten"], ["else_out"])]
    then_outputs = [
        helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [3])
    ]
    else_outputs = [
        helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [3])
    ]
    if num_outputs == 2:
        then_nodes.append(helper.make_node("Neg", ["then_out"], ["then_out2"]))
        else_nodes.append(helper.make_node("Neg", ["else_out"], ["else_out2"]))
        then_outputs.append(
            helper.make_tensor_value_info("then_out2", onnx.TensorProto.FLOAT, [3])
        )
        else_outputs.append(
            helper.make_tensor_value_info("else_out2", onnx.TensorProto.FLOAT, [3])
        )

    initializers = [
        numpy_helper.from_array(np.array([2.0] * 3, dtype=np.float32), "two"),
        numpy_helper.from_array(np.array([10.0] * 3, dtype=np.float32), "ten"),
    ]
    then_branch = helper.make_graph(then_nodes, "then_body", [], then_outputs)
    else_branch = helper.make_graph(else_nodes, "else_body", [], else_outputs)

    output_names = ["y", "z"][:num_outputs]
    node = helper.make_node(
        "If",
        ["cond"],
        output_names,
        then_branch=then_branch,
        else_branch=else_branch,
    )
    graph = helper.make_graph(
        [node],
        "if_test",
        [
            helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3]),
        ],
        [
            helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [3])
            for name in output_names
        ],
        initializer=initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )


@pytest.mark.parametrize("opset_version", [11, 13, 16, 19, 21])
@pytest.mark.parametrize("cond", [True, False])
def test_if_convert_model(opset_version, cond):
    """If must run end-to-end through ConvertModel, not only as a bare module."""
    model = make_if_model(opset_version)
    inputs = {
        "cond": np.array(cond),
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    assert_matches_oracle(model, inputs)


@pytest.mark.parametrize("cond", [True, False])
def test_if_convert_model_two_outputs(cond):
    model = make_if_model(13, num_outputs=2)
    inputs = {
        "cond": np.array(cond),
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    assert_matches_oracle(model, inputs)
