import torch
from torch import nn


class Clip(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, input, min=None, max=None):
        if min is None:
            min = self.min
        if max is None:
            max = self.max
        if min is None and max is None:
            return input
        else:
            return torch.clamp(input, min=min, max=max)
