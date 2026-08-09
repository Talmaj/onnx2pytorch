import numpy as np
import torch
import pytest

from onnx2pytorch.operations.topk import TopK
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


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


@pytest.mark.parametrize("largest", [0, 1])
@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
@pytest.mark.parametrize("opset_version", [11, 21])
def test_topk_attributes_reach_the_operator(opset_version, axis, largest):
    """axis, largest and sorted used to be dropped on the floor."""
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    k = np.array([2], dtype=np.int64)
    model = make_single_node_model(
        "TopK",
        {"x": x},
        opset_version,
        outputs=("values", "indices"),
        initializers={"k": k},
        axis=axis,
        largest=largest,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("axis", [-1, 1])
def test_topk_axis_opset_10(axis):
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    k = np.array([2], dtype=np.int64)
    model = make_single_node_model(
        "TopK",
        {"x": x},
        10,
        outputs=("values", "indices"),
        initializers={"k": k},
        axis=axis,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_topk_k_as_attribute(axis):
    """At opset 1 k is an attribute rather than an input."""
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "TopK", {"x": x}, 1, outputs=("values", "indices"), axis=axis, k=2
    )
    assert_matches_oracle(model, {"x": x})
