import pytest
import torch

from onnx2pytorch.operations import ReduceMax


@pytest.fixture
def tensor():
    return torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_reduce_max_with_dim(tensor):
    reduce_max = ReduceMax(dim=0, keepdim=True)
    output = reduce_max(tensor)
    expected_output = torch.tensor([[7, 8, 9]])

    assert output.ndim == tensor.ndim
    assert torch.equal(output, expected_output)


def test_reduce_max(tensor):
    reduce_max = ReduceMax(keepdim=False)
    output = reduce_max(tensor)
    expected_output = torch.tensor(9)

    assert output.ndim == 0
    assert torch.equal(output, expected_output)
