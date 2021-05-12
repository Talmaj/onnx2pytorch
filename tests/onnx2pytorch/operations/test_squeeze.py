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
def test_squeeze_v11(inp, dim, exp_shape):
    op = Squeeze(opset_version=11, dim=dim)
    assert tuple(op(inp).shape) == exp_shape


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
def test_squeeze_v13(inp, dim, exp_shape):
    op = Squeeze(opset_version=13)
    assert tuple(op(inp, dim).shape) == exp_shape
