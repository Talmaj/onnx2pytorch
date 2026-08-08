import torch
from torch import nn
from torch.nn import functional as F


class LpNormalization(nn.Module):
    def __init__(self, dim=-1, p=2):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, input: torch.Tensor):
        return F.normalize(input, p=float(self.p), dim=self.dim, eps=0)

    def extra_repr(self) -> str:
        return "dim={}, p={}".format(self.dim, self.p)
