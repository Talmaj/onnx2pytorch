import numpy as np
import pytest
import torch

from onnx2pytorch.operations.clip import Clip


def test_clip():
    x_np = np.random.randn(3, 4, 5).astype(np.float32)
    x = torch.from_numpy(x_np)

    op = Clip(min=-1, max=1)
    exp_y_np = np.clip(x_np, -1, 1)
    exp_y = torch.from_numpy(exp_y_np)
    assert torch.equal(op(x), exp_y)

    op = Clip(min=0)
    exp_y_np = np.clip(x_np, 0, np.inf)
    exp_y = torch.from_numpy(exp_y_np)
    assert torch.equal(op(x), exp_y)

    op = Clip(max=0)
    exp_y_np = np.clip(x_np, np.NINF, 0)
    exp_y = torch.from_numpy(exp_y_np)
    assert torch.equal(op(x), exp_y)

    op = Clip()
    exp_y_np = np.clip(x_np, np.NINF, np.inf)
    exp_y = torch.from_numpy(exp_y_np)
    assert torch.equal(op(x), exp_y)
