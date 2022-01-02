import torch
from torch import nn


class ReduceMax(nn.Module):
    def __init__(self, dim=None, keepdim=True):
        self.dim = dim
        self.keepdim = keepdim
        super().__init__()

    def forward(self, data: torch.Tensor):
        dim = self.dim
        if dim is None:
            dim = tuple(range(data.ndim))
        return torch.amax(data, dim=dim, keepdim=self.keepdim)
