import torch
from torch import nn


class Div(nn.Module):
    def forward(self, input, other):
        assert input.dtype == other.dtype
        raw_div = torch.true_divide(input, other)
        return raw_div.type(input.dtype)
