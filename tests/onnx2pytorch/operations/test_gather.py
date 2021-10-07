import numpy as np
import pytest
import torch

from onnx2pytorch.operations.gather import Gather


def test_gather_0():
    op = Gather(dim=0)
    data = torch.randn(5, 4, 3, 2, dtype=torch.float32)
    indices = torch.tensor([0, 1, 3], dtype=torch.int64)
    output_np = np.take(data.detach().numpy(), indices.detach().numpy(), axis=0)
    exp_output = torch.from_numpy(output_np).to(dtype=torch.float32)
    assert torch.equal(op(data, indices), exp_output)


def test_gather_1():
    op = Gather(dim=1)
    data = torch.randn(5, 4, 3, 2, dtype=torch.float32)
    indices = torch.tensor([0, 1, 3], dtype=torch.int64)
    output_np = np.take(data.detach().numpy(), indices.detach().numpy(), axis=1)
    exp_output = torch.from_numpy(output_np).to(dtype=torch.float32)
    assert torch.equal(op(data, indices), exp_output)


def test_gather_2d_indices():
    op = Gather(dim=1)
    data = torch.randn(3, 3, dtype=torch.float32)
    indices = torch.tensor([[0, 2]])
    output_np = np.take(data.detach().numpy(), indices.detach().numpy(), axis=1)
    exp_output = torch.from_numpy(output_np).to(dtype=torch.float32)
    assert torch.equal(op(data, indices), exp_output)
