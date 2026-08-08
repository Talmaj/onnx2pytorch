import torch
from torch import nn


class SpaceToDepth(nn.Module):
    def __init__(self, blocksize):
        super().__init__()
        self.blocksize = blocksize

    def forward(self, input: torch.Tensor):
        batch, channels, height, width = input.shape
        block = self.blocksize
        out = input.reshape(
            batch, channels, height // block, block, width // block, block
        )
        out = out.permute(0, 3, 5, 1, 2, 4)
        return out.reshape(batch, channels * block**2, height // block, width // block)

    def extra_repr(self) -> str:
        return "blocksize={}".format(self.blocksize)
