import torch
from torch import nn


class Expand(nn.Module):
    def forward(self, input: torch.Tensor, shape: torch.Tensor):
        if isinstance(shape, torch.Tensor):
            shape = shape.to(torch.int64)
        return input * torch.ones(
            torch.Size(shape), dtype=input.dtype, device=input.device
        )
