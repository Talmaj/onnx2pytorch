import torch
import pytest

from onnx2pytorch.operations.scatter import Scatter


def test_scatter_with_axis():
    op = Scatter(dim=1)
    data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    indices = torch.tensor([[1, 3]], dtype=torch.int64)
    updates = torch.tensor([[1.1, 2.1]], dtype=torch.float32)
    exp_output = torch.tensor([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=torch.float32)
    assert torch.equal(op(data, indices, updates), exp_output)


def test_scatter_without_axis():
    op = Scatter()
    data = torch.zeros((3, 3), dtype=torch.float32)
    indices = torch.tensor([[1, 0, 2], [0, 2, 1]], dtype=torch.int64)
    updates = torch.tensor([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=torch.float32)
    exp_output = torch.tensor(
        [[2.0, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]], dtype=torch.float32
    )
    assert torch.equal(op(data, indices, updates), exp_output)
