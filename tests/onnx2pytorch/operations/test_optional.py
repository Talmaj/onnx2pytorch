import torch

from onnx2pytorch.operations.optional import Optional


def test_optional_with_tensor():
    """Test Optional with a tensor input (non-empty optional)."""
    op = Optional()
    x = torch.randn(2, 3, dtype=torch.float32)

    result = op(x)

    assert result is not None
    assert torch.equal(result, x)


def test_optional_empty():
    """Test Optional without input (empty optional)."""
    op = Optional()

    result = op()

    assert result is None


def test_optional_with_none():
    """Test Optional with explicit None input (empty optional)."""
    op = Optional()

    result = op(None)

    assert result is None


def test_optional_with_sequence():
    """Test Optional with a sequence (list) input."""
    op = Optional()
    x = [torch.randn(2, 3), torch.randn(4, 5)]

    result = op(x)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert torch.equal(result[0], x[0])
    assert torch.equal(result[1], x[1])


def test_optional_with_type_attribute():
    """Test Optional accepts type attribute (for ONNX compatibility)."""
    # type attribute is accepted but not used in PyTorch
    op = Optional(type="tensor(float)")

    x = torch.randn(2, 3, dtype=torch.float32)
    result = op(x)
    assert torch.equal(result, x)

    result_empty = op()
    assert result_empty is None
