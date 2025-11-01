import numpy as np
import torch
import pytest
from onnx2pytorch.operations.autopad import AutoPad


def test_autopad_same_upper_2d():
    """Test SAME_UPPER padding for 2D input."""
    # Create AutoPad with kernel_size=3, stride=1
    autopad = AutoPad(kernel_size=3, stride=1, dilation=1, mode="SAME_UPPER")

    # Input: (batch=1, channels=1, height=5, width=5)
    x = torch.randn(1, 1, 5, 5)
    output = autopad(x)

    # With SAME padding, output should have shape (1, 1, 7, 7)
    # Total padding needed = (5-1)*1 + 3 - 5 = 2
    # SAME_UPPER: pad_before=1, pad_after=1 for each dim
    # Actually for stride=1 and kernel=3, output size = input size with SAME padding
    # So total_pad = 2, split as (1, 1) for SAME_UPPER
    assert output.shape == (1, 1, 7, 7)


def test_autopad_same_lower_2d():
    """Test SAME_LOWER padding for 2D input."""
    autopad = AutoPad(kernel_size=3, stride=1, dilation=1, mode="SAME_LOWER")

    x = torch.randn(1, 1, 5, 5)
    output = autopad(x)

    # Should have same output shape as SAME_UPPER for symmetric kernel
    assert output.shape == (1, 1, 7, 7)


def test_autopad_valid():
    """Test VALID padding (no padding)."""
    autopad = AutoPad(kernel_size=3, stride=1, dilation=1, mode="VALID")

    x = torch.randn(1, 1, 5, 5)
    output = autopad(x)

    # VALID means no padding
    assert output.shape == x.shape


def test_autopad_with_stride():
    """Test auto padding with stride > 1."""
    autopad = AutoPad(kernel_size=3, stride=2, dilation=1, mode="SAME_UPPER")

    x = torch.randn(1, 1, 6, 6)
    output = autopad(x)

    # With stride=2, output size should be ceil(6/2) = 3
    # Total padding = (3-1)*2 + 3 - 6 = 1
    # SAME_UPPER: pad_before=0, pad_after=1
    # So padded input = (1, 1, 7, 7)
    assert output.shape == (1, 1, 7, 7)


def test_autopad_with_dilation():
    """Test auto padding with dilation > 1."""
    autopad = AutoPad(kernel_size=3, stride=1, dilation=2, mode="SAME_UPPER")

    x = torch.randn(1, 1, 5, 5)
    output = autopad(x)

    # Effective kernel size = (3-1)*2 + 1 = 5
    # Total padding = (5-1)*1 + 5 - 5 = 4
    # SAME_UPPER: pad_before=2, pad_after=2
    assert output.shape == (1, 1, 9, 9)


def test_autopad_asymmetric_kernel():
    """Test with asymmetric kernel and stride."""
    autopad = AutoPad(
        kernel_size=(3, 5), stride=(1, 2), dilation=(1, 1), mode="SAME_UPPER"
    )

    x = torch.randn(1, 1, 10, 10)
    output = autopad(x)

    # Height: kernel=3, stride=1 -> output=10, total_pad = 0*1 + 3 - 10 = max(0, -7) = 0? No wait...
    # For SAME: output_size = ceil(input_size / stride)
    # Height: output = ceil(10/1) = 10, total_pad = (10-1)*1 + 3 - 10 = 2
    # Width: output = ceil(10/2) = 5, total_pad = (5-1)*2 + 5 - 10 = 3

    # So padded shape should be (1, 1, 12, 13)
    assert output.shape == (1, 1, 12, 13)


def test_autopad_1d():
    """Test auto padding for 1D input."""
    autopad = AutoPad(kernel_size=3, stride=1, dilation=1, mode="SAME_UPPER")

    # Input: (batch=1, channels=1, length=10)
    x = torch.randn(1, 1, 10)
    output = autopad(x)

    # Total padding = 2, split as (1, 1)
    assert output.shape == (1, 1, 12)


def test_autopad_invalid_mode():
    """Test that invalid mode raises error."""
    with pytest.raises(ValueError, match="Unsupported auto_pad mode"):
        AutoPad(kernel_size=3, stride=1, dilation=1, mode="INVALID")
