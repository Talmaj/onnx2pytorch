import torch
import pytest

from onnx2pytorch.operations.squeeze import Squeeze


@pytest.fixture
def inp():
    return torch.ones(1, 2, 1, 2)


@pytest.mark.parametrize(
    "dim, exp_shape",
    [
        (None, (2, 2)),
        (0, (2, 1, 2)),
        (2, (1, 2, 2)),
        (-2, (1, 2, 2)),
        (torch.tensor([0, 2]), (2, 2)),
    ],
)
def test_squeeze(inp, dim, exp_shape):
    op = Squeeze(dim)
    assert tuple(op(inp).shape) == exp_shape
