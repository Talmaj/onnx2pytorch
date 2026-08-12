import torch
from torch import nn


class Flatten(nn.Module):
    """ONNX Flatten always collapses the input into a 2D tensor around axis."""

    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, input: torch.Tensor):
        axis = self.start_dim
        if axis < 0:
            axis += input.ndim
        rows = 1
        for size in input.shape[:axis]:
            rows *= size
        columns = 1
        for size in input.shape[axis:]:
            columns *= size
        # Both sides are spelled out, reshape cannot infer one of them when the
        # other is zero.
        return input.reshape(rows, columns)

    def extra_repr(self) -> str:
        return "start_dim={}".format(self.start_dim)
