import numpy as np
import torch
from onnx2pytorch.operations.layernorm import LayerNorm


def layernorm_reference(X, scale, bias, axis=-1, epsilon=1e-5):
    """
    Reference implementation of ONNX LayerNormalization.

    Args:
        X: Input tensor
        scale: Scale tensor (gamma)
        bias: Bias tensor (beta), can be None
        axis: The first normalization dimension
        epsilon: Small value to avoid division by zero

    Returns:
        Normalized output
    """
    # Handle negative axis
    if axis < 0:
        axis = len(X.shape) + axis

    # Compute mean and variance over dimensions [axis, ..., rank-1]
    axes = tuple(range(axis, len(X.shape)))
    mean = np.mean(X, axis=axes, keepdims=True)
    variance = np.var(X, axis=axes, keepdims=True)

    # Normalize
    normalized = (X - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    Y = normalized * scale
    if bias is not None:
        Y = Y + bias

    return Y


def test_layernorm_basic():
    """Test basic LayerNormalization with default axis=-1."""
    # Input shape: (2, 3, 4)
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)

    # Expected output using reference implementation
    Y_expected = layernorm_reference(X, scale, bias, axis=-1)

    # PyTorch implementation
    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[4], eps=1e-5, axis=-1)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )


def test_layernorm_axis_minus_2():
    """Test LayerNormalization with axis=-2 (normalize over last 2 dims)."""
    # Input shape: (2, 3, 4)
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(3, 4).astype(np.float32)
    bias = np.random.randn(3, 4).astype(np.float32)

    # Expected output
    Y_expected = layernorm_reference(X, scale, bias, axis=-2)

    # PyTorch implementation
    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[3, 4], eps=1e-5, axis=-2)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )


def test_layernorm_without_bias():
    """Test LayerNormalization without bias."""
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)

    # Expected output (no bias)
    Y_expected = layernorm_reference(X, scale, None, axis=-1)

    # PyTorch implementation
    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)

    op = LayerNorm(normalized_shape=[4], eps=1e-5, axis=-1)
    Y_torch = op(X_torch, scale_torch, None)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )


def test_layernorm_2d_input():
    """Test LayerNormalization with 2D input."""
    X = np.random.randn(5, 10).astype(np.float32)
    scale = np.random.randn(10).astype(np.float32)
    bias = np.random.randn(10).astype(np.float32)

    Y_expected = layernorm_reference(X, scale, bias, axis=-1)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[10], eps=1e-5, axis=-1)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )


def test_layernorm_axis_0():
    """Test LayerNormalization with axis=0 (normalize over all dims)."""
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(2, 3, 4).astype(np.float32)
    bias = np.random.randn(2, 3, 4).astype(np.float32)

    Y_expected = layernorm_reference(X, scale, bias, axis=0)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[2, 3, 4], eps=1e-5, axis=0)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )


def test_layernorm_custom_epsilon():
    """Test LayerNormalization with custom epsilon."""
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    epsilon = 1e-3

    Y_expected = layernorm_reference(X, scale, bias, axis=-1, epsilon=epsilon)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[4], eps=epsilon, axis=-1)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-4, atol=1e-5
    )


def test_layernorm_4d_input():
    """Test LayerNormalization with 4D input (like Conv output)."""
    # Shape: (batch, channels, height, width)
    X = np.random.randn(2, 16, 8, 8).astype(np.float32)
    scale = np.random.randn(8, 8).astype(np.float32)
    bias = np.random.randn(8, 8).astype(np.float32)

    # Normalize over last 2 dimensions (spatial)
    Y_expected = layernorm_reference(X, scale, bias, axis=-2)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[8, 8], eps=1e-5, axis=-2)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )


def test_layernorm_official_onnx():
    """Test based on ONNX official example."""
    # From ONNX documentation example
    np.random.seed(0)
    X = np.random.randn(2, 3, 4, 5).astype(np.float32)
    scale = np.random.randn(4, 5).astype(np.float32)
    bias = np.random.randn(4, 5).astype(np.float32)

    # axis=-2 means normalize over dimensions [2, 3] (last 2 dims)
    axis = -2
    epsilon = 1e-5

    Y_expected = layernorm_reference(X, scale, bias, axis=axis, epsilon=epsilon)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[4, 5], eps=epsilon, axis=axis)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )


def test_layernorm_single_element_normalization():
    """Test LayerNormalization when normalizing over a single dimension."""
    X = np.random.randn(5, 10).astype(np.float32)
    scale = np.random.randn(10).astype(np.float32)
    bias = np.random.randn(10).astype(np.float32)

    Y_expected = layernorm_reference(X, scale, bias, axis=1)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[10], eps=1e-5, axis=1)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )


def test_layernorm_axis_parameter_actually_matters():
    """
    Test that proves the axis parameter is actually used.
    This would FAIL if axis is ignored and only normalized_shape is used.
    """
    X = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(3, 4).astype(np.float32)
    bias = np.random.randn(3, 4).astype(np.float32)

    # Test with axis=-2: should normalize over last 2 dims [3, 4]
    Y_expected_axis_minus2 = layernorm_reference(X, scale, bias, axis=-2)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    # Create op with wrong normalized_shape but correct axis
    # If axis is used correctly, it should still work
    op = LayerNorm(normalized_shape=[999, 999], eps=1e-5, axis=-2)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected_axis_minus2, rtol=1e-5, atol=1e-6
    )

    # Now test with axis=-1: should normalize over last dim [4] only
    scale_single = np.random.randn(4).astype(np.float32)
    bias_single = np.random.randn(4).astype(np.float32)
    Y_expected_axis_minus1 = layernorm_reference(X, scale_single, bias_single, axis=-1)

    scale_single_torch = torch.from_numpy(scale_single)
    bias_single_torch = torch.from_numpy(bias_single)

    op2 = LayerNorm(normalized_shape=[999], eps=1e-5, axis=-1)
    Y_torch2 = op2(X_torch, scale_single_torch, bias_single_torch)

    np.testing.assert_allclose(
        Y_torch2.detach().numpy(), Y_expected_axis_minus1, rtol=1e-5, atol=1e-6
    )

    # The two outputs should be DIFFERENT because they normalize over different dims
    assert not np.allclose(Y_torch.detach().numpy(), Y_torch2.detach().numpy())


def test_layernorm_negative_axis_equivalence():
    """Test that negative axis values are equivalent to their positive counterparts."""
    X = np.random.randn(2, 3, 4, 5).astype(np.float32)
    scale = np.random.randn(4, 5).astype(np.float32)
    bias = np.random.randn(4, 5).astype(np.float32)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    # axis=-2 should be equivalent to axis=2 for 4D tensor
    op_negative = LayerNorm(normalized_shape=[4, 5], eps=1e-5, axis=-2)
    Y_negative = op_negative(X_torch, scale_torch, bias_torch)

    op_positive = LayerNorm(normalized_shape=[4, 5], eps=1e-5, axis=2)
    Y_positive = op_positive(X_torch, scale_torch, bias_torch)

    np.testing.assert_allclose(
        Y_negative.detach().numpy(), Y_positive.detach().numpy(), rtol=1e-7, atol=1e-8
    )

    # axis=-1 should be equivalent to axis=3 for 4D tensor
    scale_single = np.random.randn(5).astype(np.float32)
    bias_single = np.random.randn(5).astype(np.float32)
    scale_single_torch = torch.from_numpy(scale_single)
    bias_single_torch = torch.from_numpy(bias_single)

    op_neg1 = LayerNorm(normalized_shape=[5], eps=1e-5, axis=-1)
    Y_neg1 = op_neg1(X_torch, scale_single_torch, bias_single_torch)

    op_pos3 = LayerNorm(normalized_shape=[5], eps=1e-5, axis=3)
    Y_pos3 = op_pos3(X_torch, scale_single_torch, bias_single_torch)

    np.testing.assert_allclose(
        Y_neg1.detach().numpy(), Y_pos3.detach().numpy(), rtol=1e-7, atol=1e-8
    )


def test_layernorm_transformer_style():
    """Test LayerNormalization in transformer style (batch, seq_len, hidden_dim)."""
    batch_size = 4
    seq_len = 16
    hidden_dim = 512

    X = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    scale = np.random.randn(hidden_dim).astype(np.float32)
    bias = np.random.randn(hidden_dim).astype(np.float32)

    # Normalize over the last dimension (hidden_dim)
    Y_expected = layernorm_reference(X, scale, bias, axis=-1)

    X_torch = torch.from_numpy(X)
    scale_torch = torch.from_numpy(scale)
    bias_torch = torch.from_numpy(bias)

    op = LayerNorm(normalized_shape=[hidden_dim], eps=1e-5, axis=-1)
    Y_torch = op(X_torch, scale_torch, bias_torch)

    assert Y_torch.shape == X.shape
    np.testing.assert_allclose(
        Y_torch.detach().numpy(), Y_expected, rtol=1e-5, atol=1e-6
    )
