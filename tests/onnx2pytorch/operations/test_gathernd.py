import pytest
import torch

from onnx2pytorch.operations.gathernd import GatherND


def test_gathernd_float32():
    op = GatherND(batch_dims=0)
    data = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.float32)
    indices = torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.int64)
    exp_output = torch.tensor([[[2, 3]], [[4, 5]]], dtype=torch.float32)
    assert torch.equal(op(data, indices), exp_output)


def test_gathernd_int32():
    op = GatherND(batch_dims=0)
    data = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
    exp_output = torch.tensor([0, 3], dtype=torch.int32)
    assert torch.equal(op(data, indices), exp_output)


@pytest.mark.skip("GatherND batch_dims > 0 not implemented yet")
def test_gathernd_int32_batch_dim1():
    op = GatherND(batch_dims=1)
    data = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int32)
    indices = torch.tensor([[1], [0]], dtype=torch.int64)
    exp_output = torch.tensor([[2, 3], [4, 5]], dtype=torch.int32)
    assert torch.equal(op(data, indices), exp_output)
