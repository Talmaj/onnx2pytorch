import torch
import pytest

from onnx2pytorch.operations.div import Div


def test_div():
    op = Div()
    x = torch.tensor([3, 4], dtype=torch.float32)
    y = torch.tensor([1, 2], dtype=torch.float32)
    z = x / y
    assert torch.equal(op(x, y), z)

    x = torch.randn(3, 4, 5)
    y = torch.rand(3, 4, 5) + 1.0
    z = x / y
    assert torch.equal(op(x, y), z)

    x = torch.randint(24, size=(3, 4, 5), dtype=torch.uint8)
    y = torch.randint(24, size=(3, 4, 5), dtype=torch.uint8) + 1
    z = x // y
    assert torch.equal(op(x, y), z)


def test_div_broadcast():
    op = Div()
    x = torch.randn(3, 4, 5, dtype=torch.float32)
    y = torch.rand(5, dtype=torch.float32) + 1.0
    z = x / y
    assert torch.equal(op(x, y), z)
