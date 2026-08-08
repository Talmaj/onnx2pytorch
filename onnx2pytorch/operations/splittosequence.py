import torch
from torch import nn


class SplitToSequence(nn.Module):
    def __init__(self, dim=0, keepdim=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: torch.Tensor, split: torch.Tensor = None):
        if split is None:
            # Split into single-slice chunks, keepdim decides whether to squeeze.
            chunks = list(torch.split(input, 1, dim=self.dim))
            if not self.keepdim:
                chunks = [chunk.squeeze(self.dim) for chunk in chunks]
            return chunks
        if split.dim() == 0:
            return list(torch.split(input, int(split), dim=self.dim))
        return list(torch.split(input, [int(s) for s in split], dim=self.dim))

    def extra_repr(self) -> str:
        return "dim={}, keepdim={}".format(self.dim, self.keepdim)
