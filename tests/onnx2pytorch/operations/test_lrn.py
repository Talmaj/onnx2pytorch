import math
import numpy as np
import pytest
import torch

from onnx2pytorch.operations.lrn import LRN


def lrn_reference(x, size, alpha=0.0001, beta=0.75, bias=1.0):
    """
    Reference implementation of LRN for testing.

    This matches the official ONNX implementation:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN

    Formula: y = x / ((bias + (alpha / size) * square_sum) ** beta)

    where square_sum is the sum of squares over the channel neighborhood.
    """
    n, c, h, w = x.shape
    square_sum = np.zeros((n, c, h, w)).astype(np.float32)

    for n_idx, c_idx, h_idx, w_idx in np.ndindex(x.shape):
        # Calculate the range of channels to sum over
        # Using the exact formula from ONNX reference
        c_start = max(0, c_idx - math.floor((size - 1) / 2))
        c_end = min(c, c_idx + math.ceil((size - 1) / 2) + 1)

        # Sum of squares over the channel neighborhood
        square_sum[n_idx, c_idx, h_idx, w_idx] = np.sum(
            x[n_idx, c_start:c_end, h_idx, w_idx] ** 2
        )

    # Apply LRN formula
    y = x / ((bias + (alpha / size) * square_sum) ** beta)

    return y


def test_lrn_basic():
    """Test basic LRN operation with default parameters."""
    size = 5
    alpha = 0.0001
    beta = 0.75
    bias = 1.0

    # Create a simple input
    x_np = np.random.randn(1, 10, 4, 4).astype(np.float32)
    x = torch.tensor(x_np)

    # Compute reference output
    exp_y = lrn_reference(x_np, size, alpha, beta, bias)

    # Compute LRN output
    op = LRN(size=size, alpha=alpha, beta=beta, bias=bias)
    y = op(x)

    assert y.shape == x.shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_lrn_custom_parameters():
    """Test LRN with custom parameters."""
    size = 3
    alpha = 0.001
    beta = 0.5
    bias = 2.0

    # Create a simple input
    x_np = np.random.randn(2, 8, 3, 3).astype(np.float32)
    x = torch.tensor(x_np)

    # Compute reference output
    exp_y = lrn_reference(x_np, size, alpha, beta, bias)

    # Compute LRN output
    op = LRN(size=size, alpha=alpha, beta=beta, bias=bias)
    y = op(x)

    assert y.shape == x.shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-4, atol=1e-5)


def test_lrn_size_one():
    """Test LRN with size=1 (only normalizes by the same channel)."""
    size = 1
    alpha = 0.0001
    beta = 0.75
    bias = 1.0

    x_np = np.random.randn(1, 5, 2, 2).astype(np.float32)
    x = torch.tensor(x_np)

    # Compute reference output
    exp_y = lrn_reference(x_np, size, alpha, beta, bias)

    # Compute LRN output
    op = LRN(size=size, alpha=alpha, beta=beta, bias=bias)
    y = op(x)

    assert y.shape == x.shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_lrn_large_size():
    """Test LRN with size larger than number of channels."""
    size = 20
    alpha = 0.0001
    beta = 0.75
    bias = 1.0

    # Only 10 channels but size is 20
    x_np = np.random.randn(1, 10, 3, 3).astype(np.float32)
    x = torch.tensor(x_np)

    # Compute reference output
    exp_y = lrn_reference(x_np, size, alpha, beta, bias)

    # Compute LRN output
    op = LRN(size=size, alpha=alpha, beta=beta, bias=bias)
    y = op(x)

    assert y.shape == x.shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_lrn_known_values():
    """Test LRN with known input/output values."""
    size = 3
    alpha = 1.0  # Use simple values for easier manual verification
    beta = 1.0
    bias = 0.0

    # Simple input: [[[1, 2], [3, 4], [5, 6]]]
    # Shape: (1, 3, 1, 2)
    x_np = np.array([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]]).astype(np.float32)
    x = torch.tensor(x_np)

    # Compute reference output
    exp_y = lrn_reference(x_np, size, alpha, beta, bias)

    # Compute LRN output
    op = LRN(size=size, alpha=alpha, beta=beta, bias=bias)
    y = op(x)

    assert y.shape == x.shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_lrn_zeros():
    """Test LRN with zero input."""
    size = 5
    x = torch.zeros(1, 10, 4, 4, dtype=torch.float32)

    op = LRN(size=size)
    y = op(x)

    assert y.shape == x.shape
    # With zero input, output should also be zero
    assert torch.allclose(y, torch.zeros_like(y))


def test_lrn_ones():
    """Test LRN with ones input."""
    size = 5
    alpha = 0.0001
    beta = 0.75
    bias = 1.0

    x_np = np.ones((1, 10, 4, 4), dtype=np.float32)
    x = torch.tensor(x_np)

    # Compute reference output
    exp_y = lrn_reference(x_np, size, alpha, beta, bias)

    # Compute LRN output
    op = LRN(size=size, alpha=alpha, beta=beta, bias=bias)
    y = op(x)

    assert y.shape == x.shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_lrn_no_size_error():
    """Test that LRN raises error when size is not provided."""
    with pytest.raises(ValueError, match="size parameter is required"):
        LRN()


def test_lrn_batch_size():
    """Test LRN with different batch sizes."""
    size = 5

    # Test with batch size > 1
    x = torch.randn(4, 8, 3, 3, dtype=torch.float32)

    op = LRN(size=size)
    y = op(x)

    assert y.shape == x.shape


def test_lrn_single_channel():
    """Test LRN with single channel input."""
    size = 3
    x_np = np.random.randn(1, 1, 5, 5).astype(np.float32)
    x = torch.tensor(x_np)

    # Compute reference output
    exp_y = lrn_reference(x_np, size)

    # Compute LRN output
    op = LRN(size=size)
    y = op(x)

    assert y.shape == x.shape
    assert np.allclose(y.detach().numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_lrn_compare_pytorch():
    """Test that our LRN matches PyTorch's LocalResponseNorm directly."""
    size = 5
    alpha = 0.0001
    beta = 0.75
    bias = 1.0

    x = torch.randn(2, 10, 4, 4, dtype=torch.float32)

    # Our implementation
    op = LRN(size=size, alpha=alpha, beta=beta, bias=bias)
    y1 = op(x)

    # PyTorch's implementation
    pytorch_lrn = torch.nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=bias)
    y2 = pytorch_lrn(x)

    # They should be identical
    assert torch.allclose(y1, y2, rtol=1e-6, atol=1e-7)


def test_lrn_official_onnx():
    """
    Official ONNX test case for LRN.
    From: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN
    """
    alpha = 0.0002
    beta = 0.5
    bias = 2.0
    nsize = 3

    # Use a fixed seed for reproducibility
    np.random.seed(0)
    x = np.random.randn(5, 5, 5, 5).astype(np.float32)

    # Compute expected output using reference implementation
    y_expected = lrn_reference(x, size=nsize, alpha=alpha, beta=beta, bias=bias)

    # Test our implementation
    x_torch = torch.tensor(x)
    op = LRN(alpha=alpha, beta=beta, bias=bias, size=nsize)
    y_torch = op(x_torch)

    assert y_torch.shape == x.shape
    assert np.allclose(y_torch.detach().numpy(), y_expected, rtol=1e-5, atol=1e-6)


def test_lrn_default_official_onnx():
    """
    Official ONNX test case for LRN with default parameters.
    From: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN
    """
    nsize = 3

    # Use a fixed seed for reproducibility
    np.random.seed(1)
    x = np.random.randn(5, 5, 5, 5).astype(np.float32)

    # Compute expected output using reference implementation with default parameters
    y_expected = lrn_reference(x, size=nsize)

    # Test our implementation (size is the only required parameter, others use defaults)
    x_torch = torch.tensor(x)
    op = LRN(size=nsize)
    y_torch = op(x_torch)

    assert y_torch.shape == x.shape
    assert np.allclose(y_torch.detach().numpy(), y_expected, rtol=1e-5, atol=1e-6)
