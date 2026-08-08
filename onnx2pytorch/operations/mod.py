import torch
from torch import nn


class Mod(nn.Module):
    def __init__(self, fmod=0):
        self.fmod = fmod
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        if self.fmod:
            return torch.fmod(A, B)
        return torch.remainder(A, B)
