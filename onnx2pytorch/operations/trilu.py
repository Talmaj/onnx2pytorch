import torch
from torch import nn


class Trilu(nn.Module):
    def __init__(self, upper=1):
        super().__init__()
        self.upper = bool(upper)

    def forward(self, input: torch.Tensor, k: torch.Tensor = None):
        diagonal = 0 if k is None else int(k)
        if self.upper:
            return torch.triu(input, diagonal)
        return torch.tril(input, diagonal)

    def extra_repr(self) -> str:
        return "upper={}".format(self.upper)
