import torch
from torch import nn


class GlobalLpPool(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, input: torch.Tensor):
        dim = tuple(range(2, input.ndimension()))
        pooled = torch.sum(torch.abs(input) ** self.p, dim=dim, keepdim=True)
        return pooled ** (1.0 / self.p)

    def extra_repr(self) -> str:
        return "p={}".format(self.p)
