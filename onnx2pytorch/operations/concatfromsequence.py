import torch
from torch import nn


class ConcatFromSequence(nn.Module):
    def __init__(self, dim, new_axis=0):
        super().__init__()
        self.dim = dim
        self.new_axis = bool(new_axis)

    def forward(self, input_sequence):
        if self.new_axis:
            return torch.stack(list(input_sequence), dim=self.dim)
        return torch.cat(list(input_sequence), dim=self.dim)

    def extra_repr(self) -> str:
        return "dim={}, new_axis={}".format(self.dim, self.new_axis)
