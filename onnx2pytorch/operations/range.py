import torch
from torch import nn


class Range(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor):
        return torch.arange(start=start, end=limit, step=delta)
