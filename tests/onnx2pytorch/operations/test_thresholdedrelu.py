import numpy as np
import torch
import pytest

from onnx2pytorch.operations.thresholdedrelu import ThresholdedRelu


def test_thresholdedrelu():
    x = torch.tensor([-1.5, 0.0, 1.2, 2.0, 2.2])

    op = ThresholdedRelu()
    exp_y = torch.tensor([0.0, 0.0, 1.2, 2.0, 2.2])
    assert torch.equal(op(x), exp_y)

    op = ThresholdedRelu(alpha=2.0)
    exp_y = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.2])
    assert torch.equal(op(x), exp_y)
