import torch
from torch import nn


class MeanVarianceNormalization(nn.Module):
    def __init__(self, dim=(0, 2, 3)):
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        self.dim = tuple(dim)

    def forward(self, input: torch.Tensor):
        mean = torch.mean(input, dim=self.dim, keepdim=True)
        std = torch.sqrt(torch.mean(input**2, dim=self.dim, keepdim=True) - mean**2)
        return (input - mean) / (std + 1e-9)

    def extra_repr(self) -> str:
        return "dim={}".format(self.dim)
