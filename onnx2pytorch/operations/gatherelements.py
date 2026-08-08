import torch
from torch import nn


class GatherElements(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, data: torch.Tensor, indices: torch.Tensor):
        indices = indices.long()
        indices = torch.where(indices < 0, indices + data.shape[self.dim], indices)
        return torch.gather(data, self.dim, indices)

    def extra_repr(self) -> str:
        return "dim={}".format(self.dim)
