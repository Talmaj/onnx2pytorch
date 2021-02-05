import torch
import pytest

from onnx2pytorch.operations import Resize, Upsample


@pytest.fixture
def inp():
    return torch.rand(1, 3, 10, 10)


@pytest.mark.parametrize(
    "scales, new_shape",
    [
        ([1, 1, 2, 2], [1, 3, 20, 20]),
        ([1, 1, 0.5, 0.5], [1, 3, 5, 5]),
    ],
)
def test_resize_scales(inp, scales, new_shape):
    op = Resize()
    out = op(inp, scales=scales)
    assert list(out.shape) == new_shape


@pytest.mark.parametrize("sizes", [[1, 3, 20, 20], [1, 3, 5, 5]])
def test_resize_sizes(inp, sizes):
    op = Resize()
    out = op(inp, sizes=sizes)
    assert list(out.shape) == sizes


def test_resize_raise_error(inp):
    op = Resize()

    # cannot scale batch and channel dimension
    with pytest.raises(NotImplementedError):
        op(inp, scales=[2, 2, 1, 1])
    with pytest.raises(NotImplementedError):
        op(inp, sizes=[2, 6, 10, 10])

    # need to define scales or sizes
    with pytest.raises(ValueError):
        op(inp)

    # need to define only scales or sizes
    with pytest.raises(ValueError):
        op(inp, scales=[1, 1, 2, 2], sizes=[1, 3, 20, 20])


@pytest.mark.parametrize(
    "scales, new_shape",
    [
        ([1, 1, 2, 2], [1, 3, 20, 20]),
        ([1, 1, 0.5, 0.5], [1, 3, 5, 5]),
    ],
)
def test_upsample(inp, scales, new_shape):
    op = Upsample()
    out = op(inp, scales)
    assert list(out.shape) == new_shape
