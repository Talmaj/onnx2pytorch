import torch
from torch import nn


class Expand(nn.Module):
    def forward(self, input: torch.Tensor, shape: torch.Tensor):
        return input * torch.ones(torch.Size(shape), dtype=input.dtype)
