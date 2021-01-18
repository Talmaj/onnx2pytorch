import torch
import pytest
import numpy as np
from onnx.backend.test.case.node.onehot import one_hot

from onnx2pytorch.operations import OneHot


@pytest.mark.parametrize("axis", [1, -2])
@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([[1, 9], [2, 4]], dtype=torch.float32),
        torch.tensor([0, 7, 8], dtype=torch.int64),
    ],
)
def test_onehot(indices, axis):
    on_value = 3
    off_value = 1
    output_type = torch.float32
    depth = torch.tensor([10], dtype=torch.float32)
    values = torch.tensor([off_value, on_value], dtype=output_type)
    y = one_hot(indices.numpy(), depth.numpy(), axis=axis, dtype=np.float32)
    y = y * (on_value - off_value) + off_value
    y = torch.from_numpy(y)

    op = OneHot(axis)
    out = op(indices, depth, values)
    assert torch.equal(y, out)
