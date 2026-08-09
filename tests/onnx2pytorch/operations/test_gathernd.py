import numpy as np
import pytest
import torch

from onnx2pytorch.operations.gathernd import GatherND
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


def test_gathernd_float32():
    op = GatherND(batch_dims=0)
    data = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.float32)
    indices = torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.int64)
    exp_output = torch.tensor([[[2, 3]], [[4, 5]]], dtype=torch.float32)
    assert torch.equal(op(data, indices), exp_output)


def test_gathernd_int32():
    op = GatherND(batch_dims=0)
    data = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
    exp_output = torch.tensor([0, 3], dtype=torch.int32)
    assert torch.equal(op(data, indices), exp_output)


def test_gathernd_int32_batch_dim1():
    op = GatherND(batch_dims=1)
    data = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int32)
    indices = torch.tensor([[1], [0]], dtype=torch.int64)
    exp_output = torch.tensor([[2, 3], [4, 5]], dtype=torch.int32)
    assert torch.equal(op(data, indices), exp_output)


@pytest.mark.parametrize("batch_dims", [0, 1, 2])
def test_gathernd_batch_dims(batch_dims):
    np.random.seed(0)
    data = np.random.randn(2, 3, 4, 5).astype(np.float32)
    shape = list(data.shape[:batch_dims]) + [2, 1]
    indices = np.random.randint(0, data.shape[batch_dims], size=shape, dtype=np.int64)
    model = make_single_node_model(
        "GatherND",
        {"data": data, "indices": indices},
        13,
        batch_dims=batch_dims,
    )
    assert_matches_oracle(model, {"data": data, "indices": indices})


def test_gathernd_negative_indices():
    np.random.seed(0)
    data = np.random.randn(3, 4).astype(np.float32)
    indices = np.array([[-1, -2], [0, 1]], dtype=np.int64)
    model = make_single_node_model("GatherND", {"data": data, "indices": indices}, 13)
    assert_matches_oracle(model, {"data": data, "indices": indices})
