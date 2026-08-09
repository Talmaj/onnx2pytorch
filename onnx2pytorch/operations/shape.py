import torch
from torch import nn


class Shape(nn.Module):
    def __init__(self, start=0, end=None):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, input: torch.Tensor):
        rank = input.ndim
        start = self.start + rank if self.start < 0 else self.start
        start = min(max(start, 0), rank)
        if self.end is None:
            end = rank
        else:
            end = self.end + rank if self.end < 0 else self.end
            end = min(max(end, 0), rank)
        return torch.tensor(input.shape[start:end], device=input.device)

    def extra_repr(self) -> str:
        return "start={}, end={}".format(self.start, self.end)
