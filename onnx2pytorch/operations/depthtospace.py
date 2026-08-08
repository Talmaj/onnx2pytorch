import torch
from torch import nn
from torch.nn import functional as F


class DepthToSpace(nn.Module):
    def __init__(self, blocksize, mode="DCR"):
        super().__init__()
        if mode not in ("DCR", "CRD"):
            raise NotImplementedError(
                "DepthToSpace mode={} not implemented.".format(mode)
            )
        self.blocksize = blocksize
        self.mode = mode

    def forward(self, input: torch.Tensor):
        if self.mode == "CRD":
            return F.pixel_shuffle(input, self.blocksize)

        batch, channels, height, width = input.shape
        block = self.blocksize
        out = input.reshape(batch, block, block, channels // (block**2), height, width)
        out = out.permute(0, 3, 4, 1, 5, 2)
        return out.reshape(batch, channels // (block**2), height * block, width * block)

    def extra_repr(self) -> str:
        return "blocksize={}, mode={}".format(self.blocksize, self.mode)
