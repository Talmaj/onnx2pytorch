import numpy as np
import pytest
import torch

from onnx2pytorch.operations.unsqueeze import Unsqueeze
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


def test_unsqueeze_negative_axes():
    op = Unsqueeze(opset_version=13)
    x = torch.randn(1, 3, 1, 5)
    axes = torch.tensor([-2], dtype=torch.int64)
    y = torch.tensor(np.expand_dims(x.detach().numpy(), axis=-2))
    assert torch.equal(op(x, axes), y)


def test_unsqueeze_unsorted_axes():
    op = Unsqueeze(opset_version=13)
    x = torch.randn(3, 4, 5)
    axes = torch.tensor([5, 4, 2], dtype=torch.int64)
    x_np = x.detach().numpy()
    y_np = np.expand_dims(x_np, axis=2)
    y_np = np.expand_dims(y_np, axis=4)
    y_np = np.expand_dims(y_np, axis=5)
    y = torch.tensor(y_np)
    assert torch.equal(op(x, axes), y)


@pytest.mark.parametrize(
    "axes",
    [
        [-1, -2],
        [-2, -1],
        [-3, -1],
        [-1, -3],
        [0, -1],
        [-4, -3, -2, -1],
        [1, -1],
        [-2],
    ],
)
def test_unsqueeze_negative_axes_count_against_the_output_rank(axes):
    """Axes used to be applied in the order given, against the growing input, so
    several negative ones landed in the wrong places."""
    np.random.seed(0)
    x = np.random.randn(3).astype(np.float32)
    axes_tensor = np.array(axes, dtype=np.int64)
    model = make_single_node_model(
        "Unsqueeze", {"x": x}, 13, initializers={"axes": axes_tensor}
    )
    assert_matches_oracle(model, {"x": x})
