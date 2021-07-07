import torch
import pytest

from onnx2pytorch.operations.globalaveragepool import GlobalAveragePool


def test_globalaveragepool_2d():
    op = GlobalAveragePool()
    x = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32)
    y = torch.tensor([[[[5]]]], dtype=torch.float32)
    assert torch.equal(op(x), y)


def test_globalaveragepool_3d():
    op = GlobalAveragePool()
    x = torch.tensor(
        [
            [
                [
                    [[1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6]],
                    [[7, 7], [8, 8], [9, 9]],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([[[[[5]]]]], dtype=torch.float32)
    assert torch.equal(op(x), y)


def test_globalaveragepool_channels():
    op = GlobalAveragePool()
    x = torch.tensor([[[[0, 0]], [[1, 1]]]], dtype=torch.float32)
    y = torch.tensor([[[[0]], [[1]]]], dtype=torch.float32)
    assert torch.equal(op(x), y)
