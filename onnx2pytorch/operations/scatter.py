import torch
from torch import nn


class Scatter(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor):
        return torch.scatter(data, self.dim, indices, updates)
