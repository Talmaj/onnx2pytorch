import numpy as np
import torch
import pytest
from onnx.backend.test.case.node.pad import pad_impl

from onnx2pytorch.operations import Pad
from onnx2pytorch.utils import extract_padding_params


@pytest.fixture
def inp():
    return torch.rand(1, 3, 10, 10)


# ONNX convention: [begin_d0, ..., begin_dN, end_d0, ..., end_dN].
# The asymmetric cases are what distinguish a correct begin/end ordering from a
# swapped one; symmetric pads give the same shape either way.
ONNX_PADS = [
    ([0, 0, 0, 1, 0, 0, 0, 1], [1, 3, 10, 12]),
    ([0, 0, 2, 1, 0, 0, 2, 1], [1, 3, 14, 12]),
    ([4, 3, 2, 1, 4, 3, 2, 1], [9, 9, 14, 12]),
    ([0, 0, 1, 2, 0, 0, 30, 40], [1, 3, 41, 52]),
    ([1, 0, 3, 0, 0, 2, 0, 4], [2, 5, 13, 14]),
]


@pytest.mark.parametrize("onnx_pads, new_shape", ONNX_PADS)
def test_pad_static(inp, onnx_pads, new_shape):
    """Padding passed at initialization is pre-converted by the converter."""
    op = Pad(padding=extract_padding_params(onnx_pads))
    out = op(inp)

    expected = pad_impl(inp.numpy(), np.array(onnx_pads), "constant", 0)
    assert list(out.shape) == new_shape
    assert np.array_equal(out.numpy(), expected)


@pytest.mark.parametrize("onnx_pads, new_shape", ONNX_PADS)
def test_pad_dynamic(inp, onnx_pads, new_shape):
    """Padding passed to forward arrives in raw ONNX convention."""
    op = Pad()
    out = op(inp, onnx_pads)

    expected = pad_impl(inp.numpy(), np.array(onnx_pads), "constant", 0)
    assert list(out.shape) == new_shape
    assert np.array_equal(out.numpy(), expected)


@pytest.mark.parametrize("onnx_pads, new_shape", ONNX_PADS)
def test_pad_dynamic_accepts_tensor(inp, onnx_pads, new_shape):
    """ONNX supplies pads as an int64 tensor, not a python list."""
    op = Pad()
    out = op(inp, torch.tensor(onnx_pads, dtype=torch.int64))
    assert list(out.shape) == new_shape


def test_pad_dynamic_constant_value(inp):
    """constant_value arrives as a 0-d tensor, which F.pad does not accept."""
    onnx_pads = [0, 0, 1, 2, 0, 0, 3, 4]
    op = Pad()
    out = op(inp, onnx_pads, torch.tensor(2.5))

    expected = pad_impl(inp.numpy(), np.array(onnx_pads), "constant", 2.5)
    assert np.allclose(out.numpy(), expected)


def test_pad_raise_error(inp):
    op = Pad()

    # padding should be passed either in init or forward
    with pytest.raises(TypeError):
        op(inp)
