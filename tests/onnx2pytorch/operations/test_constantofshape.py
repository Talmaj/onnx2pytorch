import numpy as np
import torch
import pytest

from onnx2pytorch.operations.constantofshape import ConstantOfShape


def test_constantofshape_float_ones():
    op = ConstantOfShape()
    x = torch.tensor([4, 3, 2], dtype=torch.int64)
    y = torch.ones(*x, dtype=torch.float32)
    assert torch.equal(op(x), y)


def test_constantofshape_int32_shape_zero():
    constant = torch.tensor([0], dtype=torch.int32)
    op = ConstantOfShape(constant=constant)
    x = torch.tensor([0], dtype=torch.int64)
    y = torch.zeros(*x, dtype=torch.int32)
    assert torch.equal(op(x), y)


def test_constantofshape_int32_zeros():
    constant = torch.tensor([0], dtype=torch.int32)
    op = ConstantOfShape(constant=constant)
    x = torch.tensor([10, 6], dtype=torch.int64)
    y = torch.zeros(*x, dtype=torch.int32)
    assert torch.equal(op(x), y)
