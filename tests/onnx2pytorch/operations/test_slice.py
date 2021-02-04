import torch
import pytest

from onnx2pytorch.operations import Slice


@pytest.fixture
def x():
    return torch.randn(20, 10, 5).to(torch.float32)


@pytest.mark.parametrize("init", [True, False])
def test_slice_1(x, init):
    starts = torch.tensor([0, 0], dtype=torch.int64)
    ends = torch.tensor([3, 10], dtype=torch.int64)
    axes = torch.tensor([0, 1], dtype=torch.int64)
    steps = torch.tensor([1, 1], dtype=torch.int64)
    y = x[0:3, 0:10]

    if init:
        op = Slice(axes, starts, ends, steps)
        assert torch.equal(op(x), y)
    else:
        op = Slice()
        assert torch.equal(op(x, starts, ends, axes, steps), y)


@pytest.mark.parametrize("init", [True, False])
def test_slice_2(x, init):
    starts = torch.tensor([1], dtype=torch.int64)
    ends = torch.tensor([1000], dtype=torch.int64)
    axes = torch.tensor([1], dtype=torch.int64)
    steps = torch.tensor([2], dtype=torch.int64)
    y = x[:, 1:1000:2]

    if init:
        op = Slice(axes, starts, ends, steps)
        assert torch.equal(op(x), y)
    else:
        op = Slice()
        assert torch.equal(op(x, starts, ends, axes, steps), y)


@pytest.mark.parametrize("init", [True, False])
def test_slice_default_axes(x, init):
    starts = torch.tensor([1, 2], dtype=torch.int64)
    ends = torch.tensor([9, 5], dtype=torch.int64)
    steps = torch.tensor([1, 2], dtype=torch.int64)
    y = x[1:9, 2:5:2]

    if init:
        op = Slice(starts=starts, ends=ends, steps=steps)
        assert torch.equal(op(x), y)
    else:
        op = Slice()
        assert torch.equal(op(x, starts, ends, steps=steps), y)


@pytest.mark.parametrize("init", [True, False])
def test_slice_default_steps(x, init):
    starts = torch.tensor([1], dtype=torch.int64)
    ends = torch.tensor([9], dtype=torch.int64)
    axes = torch.tensor([1], dtype=torch.int64)
    y = x[:, 1:9]

    if init:
        op = Slice(axes, starts, ends)
        assert torch.equal(op(x), y)
    else:
        op = Slice()
        assert torch.equal(op(x, starts, ends, axes), y)
