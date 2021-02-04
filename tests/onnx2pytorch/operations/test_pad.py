import torch
import pytest

from onnx2pytorch.operations import Pad


@pytest.fixture
def inp():
    return torch.rand(1, 3, 10, 10)


@pytest.mark.parametrize("init", [True, False])
@pytest.mark.parametrize(
    "pads, new_shape",
    [
        ([1, 1], [1, 3, 10, 12]),
        ([1, 1, 2, 2], [1, 3, 14, 12]),
        ([1, 1, 2, 2, 3, 3, 4, 4], [9, 9, 14, 12]),
    ],
)
def test_pad(inp, pads, new_shape, init):
    """Pass padding in initialization and in forward pass."""
    if init:
        op = Pad(padding=pads)
        out = op(inp)
    else:
        op = Pad()
        out = op(inp, pads)
    assert list(out.shape) == new_shape


def test_pad_raise_error(inp):
    op = Pad()

    # padding should be passed either in init or forward
    with pytest.raises(TypeError):
        op(inp)
