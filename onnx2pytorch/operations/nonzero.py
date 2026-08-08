import torch
from torch import nn


class NonZero(nn.Module):
    def forward(self, input: torch.Tensor):
        return torch.nonzero(input, as_tuple=False).t().contiguous()
