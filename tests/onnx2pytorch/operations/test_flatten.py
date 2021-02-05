import torch
import pytest

from onnx2pytorch.operations import Flatten


@pytest.fixture
def inp():
    return torch.rand(1, 3, 10, 10)


def test_flatten(inp):
    """Pass padding in initialization and in forward pass."""
    op = Flatten(1, 2)
    out = op(inp)
    assert list(out.shape) == [1, 30, 10]
