from onnx import TensorProto, helper
import numpy as np
import torch
import pytest

from onnx2pytorch.operations.constantofshape import ConstantOfShape
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


def test_constantofshape_default_is_float_zeros():
    op = ConstantOfShape()
    x = torch.tensor([4, 3, 2], dtype=torch.int64)
    y = torch.zeros(*x, dtype=torch.float32)
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


def test_constantofshape_scalar():
    """A shape tensor of length 0 asks for a 0-d tensor, and expand needs to be
    told that as an empty size rather than as no size at all."""
    shape = np.array([], dtype=np.int64)
    model = make_single_node_model("ConstantOfShape", {"shape": shape}, 20)
    assert_matches_oracle(model, {"shape": shape})


def test_constantofshape_scalar_with_value():
    shape = np.array([], dtype=np.int64)
    value = helper.make_tensor("value", TensorProto.INT32, [1], [7])
    model = make_single_node_model("ConstantOfShape", {"shape": shape}, 20, value=value)
    assert_matches_oracle(model, {"shape": shape})
