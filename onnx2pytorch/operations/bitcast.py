import torch
from torch import nn


class BitCast(nn.Module):
    def __init__(self, dtype):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.lower())
        self.dtype = dtype
        super().__init__()

    def forward(self, input: torch.Tensor):
        return input.view(self.dtype)

    def extra_repr(self) -> str:
        return "dtype={}".format(self.dtype)
