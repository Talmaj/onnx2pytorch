import torch
import pytest

from onnx2pytorch.operations.expand import Expand


def test_expand_dim_changed():
    op = Expand()
    inp = torch.reshape(torch.arange(0, 3, dtype=torch.float32), [3, 1])
    new_shape = [2, 1, 6]
    exp = inp * torch.ones(new_shape)
    exp_shape = (2, 3, 6)
    assert tuple(op(inp, new_shape).shape) == exp_shape
    assert torch.equal(op(inp, new_shape), exp)


def test_expand_dim_unchanged():
    op = Expand()
    inp = torch.reshape(torch.arange(0, 3, dtype=torch.int32), [3, 1])
    new_shape = [3, 4]
    exp = torch.cat([inp] * 4, dim=1)
    exp_shape = (3, 4)
    ret = op(inp, new_shape)
    assert tuple(ret.shape) == exp_shape
    assert torch.equal(ret, exp)
