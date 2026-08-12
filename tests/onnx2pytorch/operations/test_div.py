import numpy as np
import torch
import pytest

from onnx2pytorch.operations.div import Div
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


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


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64])
def test_div_signed_integers_truncate_towards_zero(dtype):
    """A negative quotient truncates, so -7 / 2 is -3 and not the floored -4."""
    a = np.array([-7, -1, 0, 7, -8, 8], dtype=dtype)
    b = np.array([2, 2, 3, 2, 3, -3], dtype=dtype)
    model = make_single_node_model("Div", {"a": a, "b": b}, 14)
    assert_matches_oracle(model, {"a": a, "b": b})
