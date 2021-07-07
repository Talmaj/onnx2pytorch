import torch
from torch import nn


class ScatterElements(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor):
        indices[indices < 0] = indices[indices < 0] + data.size(self.dim)
        return torch.scatter(data, self.dim, indices, updates)
