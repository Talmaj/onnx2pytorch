import numpy as np
import torch
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)

from onnx2pytorch.operations.scatterelements import ScatterElements


def test_scatter_elements_with_axis():
    op = ScatterElements(dim=1)
    data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    indices = torch.tensor([[1, 3]], dtype=torch.int64)
    updates = torch.tensor([[1.1, 2.1]], dtype=torch.float32)
    exp_output = torch.tensor([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=torch.float32)
    output = op(data, indices, updates)
    assert torch.equal(output, exp_output)


def test_scatter_elements_with_negative_indices():
    op = ScatterElements(dim=1)
    data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    indices = torch.tensor([[1, -3]], dtype=torch.int64)
    updates = torch.tensor([[1.1, 2.1]], dtype=torch.float32)
    exp_output = torch.tensor([[1.0, 1.1, 2.1, 4.0, 5.0]], dtype=torch.float32)
    output = op(data, indices, updates)
    assert torch.equal(output, exp_output)


def test_scatter_elements_without_axis():
    op = ScatterElements()
    data = torch.zeros((3, 3), dtype=torch.float32)
    indices = torch.tensor([[1, 0, 2], [0, 2, 1]], dtype=torch.int64)
    updates = torch.tensor([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=torch.float32)
    exp_output = torch.tensor(
        [[2.0, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]], dtype=torch.float32
    )
    output = op(data, indices, updates)
    assert torch.equal(output, exp_output)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reduction", ["none", "add", "mul", "max", "min"])
def test_scatterelements_reduction(reduction, axis):
    """The reduction attribute was not even accepted by the constructor."""
    np.random.seed(0)
    data = np.random.randn(3, 4).astype(np.float32)
    indices = np.array([[0, 1, 2, 0], [1, 1, 0, 2], [2, 0, 1, 1]], dtype=np.int64)
    if axis == 1:
        indices = indices % 4
    updates = np.random.randn(3, 4).astype(np.float32)
    inputs = {"data": data, "indices": indices, "updates": updates}
    model = make_single_node_model(
        "ScatterElements", inputs, 18, axis=axis, reduction=reduction
    )
    assert_matches_oracle(model, inputs)
