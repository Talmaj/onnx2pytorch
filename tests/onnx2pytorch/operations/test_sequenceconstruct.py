import numpy as np
import pytest
import torch

from onnx2pytorch.operations.sequenceconstruct import SequenceConstruct


def test_sequenceconstruct_single_tensor():
    """Test SequenceConstruct with a single tensor."""
    op = SequenceConstruct()
    x = torch.randn(2, 3, dtype=torch.float32)

    result = op(x)

    assert isinstance(result, list)
    assert len(result) == 1
    assert torch.equal(result[0], x)


def test_sequenceconstruct_multiple_tensors():
    """Test SequenceConstruct with multiple tensors."""
    op = SequenceConstruct()
    x1 = torch.randn(2, 3, dtype=torch.float32)
    x2 = torch.randn(4, 5, dtype=torch.float32)
    x3 = torch.randn(1, 2, dtype=torch.float32)

    result = op(x1, x2, x3)

    assert isinstance(result, list)
    assert len(result) == 3
    assert torch.equal(result[0], x1)
    assert torch.equal(result[1], x2)
    assert torch.equal(result[2], x3)


def test_sequenceconstruct_same_dtype():
    """Test that SequenceConstruct validates same dtype for all inputs."""
    op = SequenceConstruct()
    x1 = torch.randn(2, 3, dtype=torch.float32)
    x2 = torch.randn(2, 3, dtype=torch.float32)
    x3 = torch.randn(2, 3, dtype=torch.float32)

    result = op(x1, x2, x3)

    assert isinstance(result, list)
    assert len(result) == 3
    # All should have same dtype
    assert all(tensor.dtype == torch.float32 for tensor in result)


def test_sequenceconstruct_different_dtypes_error():
    """Test that SequenceConstruct raises error for different dtypes."""
    op = SequenceConstruct()
    x1 = torch.randn(2, 3, dtype=torch.float32)
    x2 = torch.randn(2, 3, dtype=torch.float64)

    with pytest.raises(TypeError, match="All input tensors must have the same dtype"):
        op(x1, x2)


def test_sequenceconstruct_empty_error():
    """Test that SequenceConstruct raises error with no inputs."""
    op = SequenceConstruct()

    with pytest.raises(
        ValueError, match="SequenceConstruct requires at least one input"
    ):
        op()


def test_sequenceconstruct_different_shapes_same_dtype():
    """Test SequenceConstruct with different shapes but same dtype."""
    op = SequenceConstruct()
    x1 = torch.randn(2, 3, dtype=torch.float32)
    x2 = torch.randn(5, 7, 9, dtype=torch.float32)
    x3 = torch.randn(1, dtype=torch.float32)

    result = op(x1, x2, x3)

    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0].shape == (2, 3)
    assert result[1].shape == (5, 7, 9)
    assert result[2].shape == (1,)


def test_sequenceconstruct_int_tensors():
    """Test SequenceConstruct with integer tensors."""
    op = SequenceConstruct()
    x1 = torch.tensor([1, 2, 3], dtype=torch.int64)
    x2 = torch.tensor([4, 5], dtype=torch.int64)

    result = op(x1, x2)

    assert isinstance(result, list)
    assert len(result) == 2
    assert torch.equal(result[0], x1)
    assert torch.equal(result[1], x2)
    assert all(tensor.dtype == torch.int64 for tensor in result)


def test_sequenceconstruct_bool_tensors():
    """Test SequenceConstruct with boolean tensors."""
    op = SequenceConstruct()
    x1 = torch.tensor([True, False, True], dtype=torch.bool)
    x2 = torch.tensor([False, False], dtype=torch.bool)

    result = op(x1, x2)

    assert isinstance(result, list)
    assert len(result) == 2
    assert torch.equal(result[0], x1)
    assert torch.equal(result[1], x2)


def test_sequenceconstruct_two_tensors():
    """
    Test SequenceConstruct with two tensors.
    Based on ONNX specification for variadic inputs.
    """
    op = SequenceConstruct()

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    result = op(x_torch, y_torch)

    assert isinstance(result, list)
    assert len(result) == 2
    assert torch.allclose(result[0], x_torch)
    assert torch.allclose(result[1], y_torch)


def test_sequenceconstruct_single_value():
    """Test SequenceConstruct with a single scalar-like tensor."""
    op = SequenceConstruct()
    x = torch.tensor(42.0, dtype=torch.float32)

    result = op(x)

    assert isinstance(result, list)
    assert len(result) == 1
    assert torch.equal(result[0], x)


def test_sequenceconstruct_large_sequence():
    """Test SequenceConstruct with many tensors."""
    op = SequenceConstruct()
    tensors = [torch.randn(2, 3, dtype=torch.float32) for _ in range(10)]

    result = op(*tensors)

    assert isinstance(result, list)
    assert len(result) == 10
    for i, tensor in enumerate(tensors):
        assert torch.equal(result[i], tensor)
