import torch
from torch import nn


class CumProd(nn.Module):
    def __init__(self, exclusive=0, reverse=0):
        self.exclusive = bool(exclusive)
        self.reverse = bool(reverse)
        super().__init__()

    def forward(self, x: torch.Tensor, axis: torch.Tensor):
        dim = int(axis)
        if self.reverse:
            x = torch.flip(x, [dim])
        if self.exclusive:
            ones = torch.ones_like(x.narrow(dim, 0, 1))
            x = torch.cat((ones, x.narrow(dim, 0, x.size(dim) - 1)), dim=dim)
        y = torch.cumprod(x, dim=dim)
        if self.reverse:
            y = torch.flip(y, [dim])
        return y
