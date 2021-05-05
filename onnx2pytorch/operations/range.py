import torch
from torch import nn


class Range(nn.Module):
    def forward(self, start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor):
        return torch.arange(start=start, end=limit, step=delta)
