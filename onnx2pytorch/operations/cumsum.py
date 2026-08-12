import torch
from torch import nn

from onnx2pytorch.utils import as_input_dtype


class CumSum(nn.Module):
    def __init__(self, exclusive=0, reverse=0):
        self.exclusive = bool(exclusive)
        self.reverse = bool(reverse)
        super().__init__()

    def forward(self, x: torch.Tensor, axis: torch.Tensor):
        dim = int(axis)
        if self.reverse:
            x = torch.flip(x, [dim])
        if self.exclusive:
            zeros = torch.zeros_like(x.narrow(dim, 0, 1))
            x = torch.cat((zeros, x.narrow(dim, 0, x.size(dim) - 1)), dim=dim)
        y = as_input_dtype(torch.cumsum(x, dim=dim), x)
        if self.reverse:
            y = torch.flip(y, [dim])
        return y
