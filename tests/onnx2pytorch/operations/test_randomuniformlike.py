import numpy as np
import pytest
import torch

from onnx2pytorch.operations.randomuniformlike import RandomUniformLike


def test_randomuniformlike_basic():
    """Test basic RandomUniformLike operation with default parameters."""
    x = torch.randn(3, 4, 5, dtype=torch.float32)

    op = RandomUniformLike()
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check dtype matches input
    assert y.dtype == x.dtype
    # Check values are in default range [0, 1)
    assert torch.all(y >= 0.0)
    assert torch.all(y < 1.0)


def test_randomuniformlike_custom_range():
    """Test RandomUniformLike with custom low and high values."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    low = -5.0
    high = 10.0

    op = RandomUniformLike(low=low, high=high)
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check values are in specified range [low, high)
    assert torch.all(y >= low)
    assert torch.all(y < high)


def test_randomuniformlike_with_seed():
    """Test that seed produces reproducible results."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    seed = 42

    # Generate with seed
    op1 = RandomUniformLike(seed=seed)
    y1 = op1(x)

    # Generate again with same seed
    op2 = RandomUniformLike(seed=seed)
    y2 = op2(x)

    # Results should be identical
    assert torch.equal(y1, y2)


def test_randomuniformlike_different_seeds():
    """Test that different seeds produce different results."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    op1 = RandomUniformLike(seed=42)
    y1 = op1(x)

    op2 = RandomUniformLike(seed=123)
    y2 = op2(x)

    # Results should be different (with very high probability)
    assert not torch.equal(y1, y2)


def test_randomuniformlike_dtype_float32():
    """Test RandomUniformLike with explicit float32 dtype."""
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    # ONNX TensorProto.FLOAT = 1
    op = RandomUniformLike(dtype=1)
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check dtype is float32, not the input dtype (float64)
    assert y.dtype == torch.float32


def test_randomuniformlike_dtype_float64():
    """Test RandomUniformLike with explicit float64 dtype."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    # ONNX TensorProto.DOUBLE = 11
    op = RandomUniformLike(dtype=11)
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check dtype is float64, not the input dtype (float32)
    assert y.dtype == torch.float64


def test_randomuniformlike_dtype_float16():
    """Test RandomUniformLike with explicit float16 dtype."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    # ONNX TensorProto.FLOAT16 = 10
    op = RandomUniformLike(dtype=10)
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check dtype is float16
    assert y.dtype == torch.float16


def test_randomuniformlike_no_dtype():
    """Test RandomUniformLike without dtype uses input dtype."""
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    op = RandomUniformLike()
    y = op(x)

    # Check dtype matches input
    assert y.dtype == x.dtype


def test_randomuniformlike_negative_range():
    """Test RandomUniformLike with negative range."""
    x = torch.randn(3, 4, dtype=torch.float32)
    low = -10.0
    high = -5.0

    op = RandomUniformLike(low=low, high=high)
    y = op(x)

    # Check values are in specified range
    assert torch.all(y >= low)
    assert torch.all(y < high)


def test_randomuniformlike_large_tensor():
    """Test RandomUniformLike with larger tensor."""
    x = torch.randn(10, 20, 30, dtype=torch.float32)

    op = RandomUniformLike(low=-1.0, high=1.0)
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check values are in specified range
    assert torch.all(y >= -1.0)
    assert torch.all(y < 1.0)


def test_randomuniformlike_1d_tensor():
    """Test RandomUniformLike with 1D tensor."""
    x = torch.randn(100, dtype=torch.float32)

    op = RandomUniformLike()
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check values are in default range
    assert torch.all(y >= 0.0)
    assert torch.all(y < 1.0)


def test_randomuniformlike_distribution_check():
    """Test that RandomUniformLike produces values with roughly uniform distribution."""
    x = torch.zeros(10000, dtype=torch.float32)
    low = 0.0
    high = 10.0

    op = RandomUniformLike(low=low, high=high, seed=42)
    y = op(x)

    # Check mean is roughly in the middle of the range
    expected_mean = (low + high) / 2.0
    actual_mean = y.mean().item()
    # Allow 5% tolerance
    assert abs(actual_mean - expected_mean) < 0.5

    # Check that values are spread across the range
    # Divide into 10 bins and check each has some values
    for i in range(10):
        bin_low = low + i * (high - low) / 10
        bin_high = low + (i + 1) * (high - low) / 10
        count = torch.sum((y >= bin_low) & (y < bin_high))
        # Each bin should have roughly 1000 values (10% of 10000)
        # Allow wide tolerance since this is a statistical test
        assert count > 500  # At least 5% in each bin


def test_randomuniformlike_zero_width_range():
    """Test RandomUniformLike with low == high."""
    x = torch.randn(2, 3, dtype=torch.float32)
    value = 5.0

    op = RandomUniformLike(low=value, high=value)
    y = op(x)

    # All values should be exactly the low/high value
    assert torch.allclose(y, torch.full_like(y, value))


def test_randomuniformlike_device_preservation():
    """Test that RandomUniformLike preserves device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(2, 3, 4, dtype=torch.float32, device="cuda")

    op = RandomUniformLike()
    y = op(x)

    # Check device is preserved
    assert y.device == x.device
    # Check shape matches
    assert y.shape == x.shape


def test_randomuniformlike_unsupported_dtype():
    """Test that RandomUniformLike raises error for unsupported dtype."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    # ONNX TensorProto.UINT16 = 4 (not supported in PyTorch)
    op = RandomUniformLike(dtype=4)

    with pytest.raises(ValueError, match="ONNX dtype .* is not supported"):
        y = op(x)


def test_randomuniformlike_dtype_int8():
    """Test RandomUniformLike with int8 dtype."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    # ONNX TensorProto.INT8 = 3
    op = RandomUniformLike(dtype=3, low=-10.0, high=10.0)
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check dtype is int8
    assert y.dtype == torch.int8


def test_randomuniformlike_dtype_bool():
    """Test RandomUniformLike with bool dtype."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    # ONNX TensorProto.BOOL = 9
    op = RandomUniformLike(dtype=9)
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check dtype is bool
    assert y.dtype == torch.bool


def test_randomuniformlike_dtype_int64():
    """Test RandomUniformLike with int64 dtype."""
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    # ONNX TensorProto.INT64 = 7
    op = RandomUniformLike(dtype=7, low=0.0, high=100.0)
    y = op(x)

    # Check shape matches input
    assert y.shape == x.shape
    # Check dtype is int64
    assert y.dtype == torch.int64
