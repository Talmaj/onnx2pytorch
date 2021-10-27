import numpy as np
import torch
import pytest

from onnx2pytorch.operations import Slice
from onnx2pytorch.operations.slice import _to_positive_step


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
def test_slice_neg_axes(x, init):
    starts = torch.tensor([1], dtype=torch.int64)
    ends = torch.tensor([4], dtype=torch.int64)
    axes = torch.tensor([-1], dtype=torch.int64)
    steps = torch.tensor([2], dtype=torch.int64)
    y = x[:, :, 1:4:2]

    if init:
        op = Slice(axes, starts, ends, steps)
        assert torch.equal(op(x), y)
    else:
        op = Slice()
        assert torch.equal(op(x, starts, ends, axes, steps), y)


@pytest.mark.parametrize("init", [True, False])
def test_slice_neg_axes_2(x, init):
    print(x.shape)
    starts = torch.tensor([1], dtype=torch.int64)
    ends = torch.tensor([4], dtype=torch.int64)
    axes = torch.tensor([-2], dtype=torch.int64)
    steps = torch.tensor([2], dtype=torch.int64)
    y = x[:, 1:4:2]

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


@pytest.mark.parametrize("init", [True, False])
def test_slice_neg_steps(x, init):
    starts = torch.tensor([20, 10, 4], dtype=torch.int64)
    ends = torch.tensor([0, 0, 1], dtype=torch.int64)
    axes = torch.tensor([0, 1, 2], dtype=torch.int64)
    steps = torch.tensor([-1, -3, -2], dtype=torch.int64)
    y = torch.tensor(np.copy(x.numpy()[20:0:-1, 10:0:-3, 4:1:-2]))

    if init:
        op = Slice(axes, starts=starts, ends=ends, steps=steps)
        print(op, flush=True)
        assert torch.equal(op(x), y)
    else:
        op = Slice()
        assert torch.equal(op(x, starts, ends, axes, steps), y)


def test_to_positive_step():
    assert _to_positive_step(slice(-1, None, -1), 8) == slice(0, 8, 1)
    assert _to_positive_step(slice(-2, None, -1), 8) == slice(0, 7, 1)
    assert _to_positive_step(slice(None, -1, -1), 8) == slice(0, 0, 1)
    assert _to_positive_step(slice(None, -2, -1), 8) == slice(7, 8, 1)
    assert _to_positive_step(slice(None, None, -1), 8) == slice(0, 8, 1)
    assert _to_positive_step(slice(8, 1, -2), 8) == slice(3, 8, 2)
    assert _to_positive_step(slice(8, 0, -2), 8) == slice(1, 8, 2)
