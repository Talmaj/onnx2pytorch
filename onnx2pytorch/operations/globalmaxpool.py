import torch
from torch import nn


class GlobalMaxPool(nn.Module):
    def forward(self, input: torch.Tensor):
        dim = tuple(range(2, input.ndimension()))
        return torch.amax(input, dim=dim, keepdim=True)
