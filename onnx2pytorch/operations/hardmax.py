import torch
from torch import nn


class Hardmax(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super().__init__()

    def forward(self, input: torch.Tensor):
        maximal = input == torch.max(input, dim=self.dim, keepdim=True).values
        # In case of ties only the first maximal element is set to 1
        first = torch.cumsum(maximal.to(torch.int64), dim=self.dim) == 1
        return torch.logical_and(maximal, first).to(input.dtype)
