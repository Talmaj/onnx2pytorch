import torch
from torch import nn


class Compress(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor, condition: torch.Tensor):
        condition = condition.flatten().bool()
        if self.dim is None:
            input = input.flatten()
            dim = 0
        else:
            dim = self.dim
        indices = torch.nonzero(condition).flatten()
        indices = indices[indices < input.shape[dim]]
        return torch.index_select(input, dim, indices)

    def extra_repr(self) -> str:
        return "dim={}".format(self.dim)
