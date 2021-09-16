import numpy as np
import pytest
import torch

from onnx2pytorch.operations.unsqueeze import Unsqueeze


def test_unsqueeze_negative_axes():
    op = Unsqueeze(opset_version=13)
    x = torch.randn(1, 3, 1, 5)
    axes = torch.tensor([-2], dtype=torch.int64)
    y = torch.from_numpy(np.expand_dims(x.detach().numpy(), axis=-2))
    assert torch.equal(op(x, axes), y)


def test_unsqueeze_unsorted_axes():
    op = Unsqueeze(opset_version=13)
    x = torch.randn(3, 4, 5)
    axes = torch.tensor([5, 4, 2], dtype=torch.int64)
    x_np = x.detach().numpy()
    y_np = np.expand_dims(x_np, axis=2)
    y_np = np.expand_dims(y_np, axis=4)
    y_np = np.expand_dims(y_np, axis=5)
    y = torch.from_numpy(y_np)
    assert torch.equal(op(x, axes), y)
