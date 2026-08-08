import torch
from torch import nn


class Shrink(nn.Module):
    def __init__(self, bias=0.0, lambd=0.5):
        self.bias = bias
        self.lambd = lambd
        super().__init__()

    def forward(self, input: torch.Tensor):
        zeros = torch.zeros_like(input)
        negative = torch.where(input < -self.lambd, input + self.bias, zeros)
        return torch.where(input > self.lambd, input - self.bias, negative)
