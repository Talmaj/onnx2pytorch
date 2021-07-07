import pytest
import torch

from onnx2pytorch.operations.prelu import PRelu


def test_prelu():
    op = PRelu()
    x = torch.randn(3, 4, 5, dtype=torch.float32)
    slope = torch.randn(3, 4, 5, dtype=torch.float32)
    exp_y = torch.maximum(torch.zeros_like(x), x) + slope * torch.minimum(
        torch.zeros_like(x), x
    )
    assert torch.equal(op(x, slope), exp_y)


def test_prelu_broadcast():
    op = PRelu()
    x = torch.randn(3, 4, 5, dtype=torch.float32)
    slope = torch.randn(5, dtype=torch.float32)
    exp_y = torch.clamp(x, min=0) + torch.clamp(x, max=0) * slope
    assert torch.equal(op(x, slope), exp_y)
