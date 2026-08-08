import torch
import pytest

from onnx2pytorch.operations import Pad


@pytest.fixture
def inp():
    return torch.rand(1, 3, 10, 10)


@pytest.mark.parametrize(
    "pads, new_shape",
    [
        # pads in PyTorch format: [left, right, top, bottom, ...]
        ([1, 1], [1, 3, 10, 12]),
        ([1, 1, 2, 2], [1, 3, 14, 12]),
        ([1, 1, 2, 2, 3, 3, 4, 4], [9, 9, 14, 12]),
    ],
)
def test_pad_static(inp, pads, new_shape):
    """Pass padding in initialization (PyTorch format, pre-converted from ONNX)."""
    op = Pad(padding=pads)
    out = op(inp)
    assert list(out.shape) == new_shape


@pytest.mark.parametrize(
    "pads, new_shape",
    [
        # pads in ONNX format: [begin_d0, ..., begin_dN, end_d0, ..., end_dN]
        # For 4D input [1, 3, 10, 10]:
        ([0, 0, 0, 1, 0, 0, 0, 1], [1, 3, 10, 12]),  # pad last dim by (1, 1)
        ([0, 0, 2, 1, 0, 0, 2, 1], [1, 3, 14, 12]),  # pad last 2 dims
        ([4, 3, 2, 1, 4, 3, 2, 1], [9, 9, 14, 12]),  # pad all dims
    ],
)
def test_pad_dynamic(inp, pads, new_shape):
    """Pass padding in forward pass (ONNX format, as from ONNX runtime inputs)."""
    op = Pad()
    out = op(inp, pads)
    assert list(out.shape) == new_shape


def test_pad_raise_error(inp):
    op = Pad()

    # padding should be passed either in init or forward
    with pytest.raises(TypeError):
        op(inp)
