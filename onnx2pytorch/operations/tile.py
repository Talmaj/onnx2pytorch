import torch
from torch import nn


class Tile(nn.Module):
    def forward(self, input: torch.Tensor, repeats: torch.Tensor):
        return torch.tile(input, tuple(repeats))
