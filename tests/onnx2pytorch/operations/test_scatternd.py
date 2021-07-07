import torch
import pytest

from onnx2pytorch.operations.scatternd import ScatterND


@pytest.mark.parametrize(
    "data, indices, updates, exp_output",
    [
        (
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            torch.tensor([[4], [3], [1], [7]]),
            torch.tensor([9, 10, 11, 12]),
            torch.tensor([1, 11, 3, 10, 9, 6, 7, 12]),
        ),
        (
            torch.zeros((4, 4, 4), dtype=torch.int64),
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([[5, 5, 5, 5], [6, 6, 6, 6]]),
            torch.tensor(
                [
                    [[0, 0, 0, 0], [5, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [6, 6, 6, 6]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                ]
            ),
            torch.tensor([[0], [2]]),
            torch.tensor(
                [
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                ]
            ),
            torch.tensor(
                [
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                ]
            ),
        ),
    ],
)
def test_scatternd(data, indices, updates, exp_output):
    op = ScatterND()
    assert torch.equal(op(data, indices, updates), exp_output)
