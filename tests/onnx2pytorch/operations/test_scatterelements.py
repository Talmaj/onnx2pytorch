import torch
import pytest

from onnx2pytorch.operations.scatterelements import ScatterElements


def test_scatter_elements_with_axis():
    op = ScatterElements(dim=1)
    data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    indices = torch.tensor([[1, 3]], dtype=torch.int64)
    updates = torch.tensor([[1.1, 2.1]], dtype=torch.float32)
    exp_output = torch.tensor([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=torch.float32)
    output = op(data, indices, updates)
    assert torch.equal(output, exp_output)


def test_scatter_elements_with_negative_indices():
    op = ScatterElements(dim=1)
    data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    indices = torch.tensor([[1, -3]], dtype=torch.int64)
    updates = torch.tensor([[1.1, 2.1]], dtype=torch.float32)
    exp_output = torch.tensor([[1.0, 1.1, 2.1, 4.0, 5.0]], dtype=torch.float32)
    output = op(data, indices, updates)
    assert torch.equal(output, exp_output)


def test_scatter_elements_without_axis():
    op = ScatterElements()
    data = torch.zeros((3, 3), dtype=torch.float32)
    indices = torch.tensor([[1, 0, 2], [0, 2, 1]], dtype=torch.int64)
    updates = torch.tensor([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=torch.float32)
    exp_output = torch.tensor(
        [[2.0, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]], dtype=torch.float32
    )
    output = op(data, indices, updates)
    assert torch.equal(output, exp_output)
