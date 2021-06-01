import torch
import pytest

from onnx2pytorch.operations.reducesum import ReduceSum


@pytest.fixture
def inp():
    return torch.tensor(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
        dtype=torch.float32,
    )


def test_reducesum_default_axes_keepdims(inp):
    op = ReduceSum(opset_version=13)
    exp = torch.tensor([[[78]]], dtype=torch.float32)
    assert torch.equal(op(inp), exp)


def test_reducesum_do_not_keepdims(inp):
    op = ReduceSum(opset_version=13, keepdim=False)
    axes = torch.tensor([1])
    exp = torch.tensor(
        [[4, 6], [12, 14], [20, 22]],
        dtype=torch.float32,
    )
    assert torch.equal(op(inp, axes), exp)


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("axes", [None, torch.tensor([])])
def test_reducesum_empty_axes_input_noop(inp, keepdim, axes):
    op = ReduceSum(opset_version=13, keepdim=keepdim, noop_with_empty_axes=True)
    exp = inp
    assert torch.equal(op(inp), exp)
