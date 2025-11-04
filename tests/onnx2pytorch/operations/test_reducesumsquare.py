import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.reducesumsquare import ReduceSumSquare


@pytest.mark.parametrize(
    "input_shape,axes,keepdims",
    [
        # Test with different axes
        ([3, 4, 5], [0], 1),
        ([3, 4, 5], [1], 1),
        ([3, 4, 5], [2], 1),
        ([3, 4, 5], [-1], 1),
        # Test with multiple axes
        ([3, 4, 5], [0, 1], 1),
        ([3, 4, 5], [1, 2], 1),
        ([3, 4, 5], [0, 2], 1),
        # Test with keepdims=0
        ([3, 4, 5], [1], 0),
        ([3, 4, 5], [0, 2], 0),
        # Test with all axes (None means reduce all)
        ([3, 4, 5], None, 1),
        ([3, 4, 5], None, 0),
        # Test 2D inputs
        ([5, 10], [0], 1),
        ([5, 10], [1], 1),
        ([5, 10], None, 1),
        # Test 1D inputs
        ([10], [0], 1),
        ([10], None, 1),
    ],
)
def test_reducesumsquare_onnxruntime(input_shape, axes, keepdims):
    """Test ReduceSumSquare against onnxruntime."""
    np.random.seed(42)

    # Create input
    X = np.random.randn(*input_shape).astype(np.float32)

    # Create ONNX graph with ReduceSumSquare node
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    # Use axes as attribute (supported in all opset versions)
    node_attrs = {"keepdims": keepdims}
    if axes is not None:
        node_attrs["axes"] = axes

    reducesumsquare_node = helper.make_node(
        "ReduceSumSquare",
        inputs=["X"],
        outputs=["Y"],
        **node_attrs,
    )

    graph = helper.make_graph(
        [reducesumsquare_node],
        "reducesumsquare_test",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=8
    )

    # Run with onnxruntime
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_outputs = ort_session.run(None, {"X": X})
    expected_Y = ort_outputs[0]

    # Convert to PyTorch and run
    o2p_model = ConvertModel(model, experimental=True)
    X_torch = torch.from_numpy(X)

    with torch.no_grad():
        o2p_output = o2p_model(X_torch)

    # Compare outputs
    torch.testing.assert_close(
        o2p_output,
        torch.from_numpy(expected_Y),
        rtol=1e-5,
        atol=1e-5,
    )


def test_reducesumsquare_formula():
    """Test that ReduceSumSquare implements sum(x^2)."""
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Manual computation
    expected_all = torch.sum(X**2)
    expected_axis0 = torch.sum(X**2, dim=0, keepdim=True)
    expected_axis1 = torch.sum(X**2, dim=1, keepdim=True)

    # Test reduce all

    op_all = ReduceSumSquare(opset_version=13, dim=None, keepdim=True)
    result_all = op_all(X)
    torch.testing.assert_close(
        result_all, expected_all.view(1, 1), rtol=1e-6, atol=1e-6
    )

    # Test reduce axis 0
    op_axis0 = ReduceSumSquare(opset_version=11, dim=0, keepdim=True)
    result_axis0 = op_axis0(X)
    torch.testing.assert_close(result_axis0, expected_axis0, rtol=1e-6, atol=1e-6)

    # Test reduce axis 1
    op_axis1 = ReduceSumSquare(opset_version=11, dim=1, keepdim=True)
    result_axis1 = op_axis1(X)
    torch.testing.assert_close(result_axis1, expected_axis1, rtol=1e-6, atol=1e-6)


def test_reducesumsquare_keepdims():
    """Test keepdims parameter."""
    X = torch.randn(2, 3, 4)

    # With keepdims=True
    op_keep = ReduceSumSquare(opset_version=11, dim=1, keepdim=True)
    result_keep = op_keep(X)
    assert result_keep.shape == (2, 1, 4)

    # With keepdims=False
    op_no_keep = ReduceSumSquare(opset_version=11, dim=1, keepdim=False)
    result_no_keep = op_no_keep(X)
    assert result_no_keep.shape == (2, 4)

    # Values should be the same (just different shapes)
    torch.testing.assert_close(
        result_keep.squeeze(1), result_no_keep, rtol=1e-6, atol=1e-6
    )


def test_reducesumsquare_noop_with_empty_axes():
    """Test noop_with_empty_axes parameter."""
    X = torch.randn(2, 3, 4)

    # With noop_with_empty_axes=True and no axes, should return input unchanged
    op_noop = ReduceSumSquare(
        opset_version=13, dim=None, keepdim=True, noop_with_empty_axes=True
    )
    result_noop = op_noop(X)
    torch.testing.assert_close(result_noop, X, rtol=1e-6, atol=1e-6)

    # With noop_with_empty_axes=False and no axes, should reduce all
    op_reduce = ReduceSumSquare(
        opset_version=13, dim=None, keepdim=True, noop_with_empty_axes=False
    )
    result_reduce = op_reduce(X)
    expected = torch.sum(X**2).view(1, 1, 1)
    torch.testing.assert_close(result_reduce, expected, rtol=1e-6, atol=1e-6)


def test_reducesumsquare_with_axes_input():
    """Test with axes as an input tensor (for frameworks that support it)."""
    X = torch.randn(2, 3, 4)

    # Opset 13+ supports axes as input
    op = ReduceSumSquare(opset_version=13, dim=None, keepdim=True)

    # Provide axes as a tensor
    axes = torch.tensor([0, 2], dtype=torch.int64)
    result = op(X, axes)

    # Expected: reduce along axes 0 and 2
    expected = torch.sum(X**2, dim=(0, 2), keepdim=True)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
    assert result.shape == (1, 3, 1)


def test_reducesumsquare_vs_reducesum_square():
    """Test that ReduceSumSquare(x) == ReduceSum(Square(x))."""
    X = torch.randn(3, 4, 5)

    # ReduceSumSquare
    op_sumsquare = ReduceSumSquare(opset_version=11, dim=1, keepdim=True)
    result_sumsquare = op_sumsquare(X)

    # ReduceSum(Square(x))
    result_square_sum = torch.sum(X**2, dim=1, keepdim=True)

    torch.testing.assert_close(
        result_sumsquare, result_square_sum, rtol=1e-6, atol=1e-6
    )


def test_reducesumsquare_negative_axis():
    """Test with negative axis values."""
    X = torch.randn(2, 3, 4)

    # axis=-1 should be equivalent to axis=2
    op_neg = ReduceSumSquare(opset_version=11, dim=-1, keepdim=True)
    result_neg = op_neg(X)

    op_pos = ReduceSumSquare(opset_version=11, dim=2, keepdim=True)
    result_pos = op_pos(X)

    torch.testing.assert_close(result_neg, result_pos, rtol=1e-6, atol=1e-6)


def test_reducesumsquare_gradient():
    """Test that gradients flow correctly through ReduceSumSquare."""
    X = torch.randn(2, 3, 4, requires_grad=True)

    op = ReduceSumSquare(opset_version=11, dim=1, keepdim=True)
    result = op(X)

    # Compute gradient
    loss = result.sum()
    loss.backward()

    # Gradient of sum(x^2) with respect to x is 2x
    # After summing along dim=1, gradient should be 2x broadcast along dim=1
    expected_grad = 2 * X

    assert X.grad is not None
    torch.testing.assert_close(X.grad, expected_grad, rtol=1e-5, atol=1e-5)
