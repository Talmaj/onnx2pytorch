import torch
from torch import nn


class Expand(nn.Module):
    def forward(self, input: torch.Tensor, shape: torch.Tensor):
        #return input.expand(torch.Size(shape))
        return input * torch.ones(torch.Size(shape), dtype=input.dtype)

