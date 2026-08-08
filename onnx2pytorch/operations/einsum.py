import torch
from torch import nn


class Einsum(nn.Module):
    def __init__(self, equation):
        self.equation = equation
        super().__init__()

    def forward(self, *inputs):
        return torch.einsum(self.equation, *inputs)
