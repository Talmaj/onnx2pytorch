import torch
from torch import nn


class ArgMin(nn.Module):
    def __init__(self, dim=0, keepdim=True, select_last_index=0):
        self.dim = dim
        self.keepdim = bool(keepdim)
        self.select_last_index = bool(select_last_index)
        super().__init__()

    def forward(self, data: torch.Tensor):
        if self.select_last_index:
            flipped = torch.flip(data, [self.dim])
            indices = torch.argmin(flipped, dim=self.dim, keepdim=self.keepdim)
            return data.size(self.dim) - 1 - indices
        return torch.argmin(data, dim=self.dim, keepdim=self.keepdim)
