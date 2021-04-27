import torch
from torch import nn


class Expand(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, shape: torch.Tensor):
        return input * torch.ones(torch.Size(shape))
