import torch
import pytest

from onnx2pytorch.operations.where import Where


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_where(dtype):
    op = Where()
    condition = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[9, 8], [7, 6]])
    z = torch.tensor([[1, 8], [3, 4]])
    assert torch.equal(op(condition, x.to(dtype), y.to(dtype)), z.to(dtype))
