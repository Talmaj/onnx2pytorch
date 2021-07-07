import torch
from torch import nn


class Transpose(nn.Module):
    def __init__(self, dims=None):
        self.dims = dims
        super().__init__()

    def forward(self, data: torch.Tensor):
        if not self.dims:
            dims = tuple(reversed(range(data.dim())))
        else:
            dims = self.dims
        transposed = data.permute(dims)
        return transposed
