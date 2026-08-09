import numpy as np
import torch
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)

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


@pytest.mark.parametrize("reduction", ["none", "add", "mul", "max", "min"])
def test_scatternd_reduction(reduction):
    """The reduction attribute was ignored, so updates always overwrote."""
    np.random.seed(0)
    data = np.random.randn(4, 4, 4).astype(np.float32)
    # Index 0 appears twice, which is what distinguishes the reductions
    indices = np.array([[0], [2], [0]], dtype=np.int64)
    updates = np.random.randn(3, 4, 4).astype(np.float32)
    inputs = {"data": data, "indices": indices, "updates": updates}
    model = make_single_node_model("ScatterND", inputs, 18, reduction=reduction)
    assert_matches_oracle(model, inputs)


def test_scatternd_nested_indices():
    np.random.seed(0)
    data = np.random.randn(4, 4, 4).astype(np.float32)
    indices = np.array([[[0, 1], [1, 2]]], dtype=np.int64)
    updates = np.random.randn(1, 2, 4).astype(np.float32)
    inputs = {"data": data, "indices": indices, "updates": updates}
    model = make_single_node_model("ScatterND", inputs, 18, reduction="add")
    assert_matches_oracle(model, inputs)
