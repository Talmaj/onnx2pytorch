import torch
import pytest

from onnx2pytorch.operations.range import Range


@pytest.mark.parametrize(
    "start, limit, delta, expected",
    [
        (3, 9, 3, torch.tensor([3, 6])),
        (10, 4, -2, torch.tensor([10, 8, 6])),
        (10, 6, -3, torch.tensor([10, 7])),
    ],
)
def test_range(start, limit, delta, expected):
    op = Range()
    assert torch.equal(op(start, limit, delta), expected)
