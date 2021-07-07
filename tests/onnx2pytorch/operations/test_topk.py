import torch
import pytest

from onnx2pytorch.operations.topk import TopK


def test_topk():
    axis = 1
    largest = 1
    op = TopK(axis=axis, largest=largest)

    X = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=torch.float32,
    )
    k = 3
    K = torch.tensor([k], dtype=torch.int64)
    values_exp = torch.tensor(
        [
            [3, 2, 1],
            [7, 6, 5],
            [11, 10, 9],
        ],
        dtype=torch.float32,
    )
    indices_exp = torch.tensor(
        [
            [3, 2, 1],
            [3, 2, 1],
            [3, 2, 1],
        ]
    )
    values, indices = op(X, K)
    assert torch.equal(values_exp, values)
    assert torch.equal(indices_exp, indices)


def test_topk_negative_axis():
    op = TopK()

    X = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=torch.float32,
    )
    k = 3
    K = torch.tensor([k], dtype=torch.int64)
    values_exp = torch.tensor(
        [
            [3, 2, 1],
            [7, 6, 5],
            [11, 10, 9],
        ],
        dtype=torch.float32,
    )
    indices_exp = torch.tensor(
        [
            [3, 2, 1],
            [3, 2, 1],
            [3, 2, 1],
        ]
    )
    values, indices = op(X, K)
    assert torch.equal(values_exp, values)
    assert torch.equal(indices_exp, indices)


def test_topk_smallest():
    axis = 1
    largest = 0
    op = TopK(axis=axis, largest=largest)

    X = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [11, 10, 9, 8],
        ],
        dtype=torch.float32,
    )
    k = 3
    K = torch.tensor([k], dtype=torch.int64)
    values_exp = torch.tensor(
        [
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
        ],
        dtype=torch.float32,
    )
    indices_exp = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 2],
            [3, 2, 1],
        ]
    )
    values, indices = op(X, K)
    assert torch.equal(values_exp, values)
    assert torch.equal(indices_exp, indices)
