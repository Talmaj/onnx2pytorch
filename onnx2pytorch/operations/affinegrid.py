import torch
from torch import nn
from torch.nn import functional as F


class AffineGrid(nn.Module):
    def __init__(self, align_corners=0):
        super().__init__()
        self.align_corners = bool(align_corners)

    def forward(self, theta: torch.Tensor, size: torch.Tensor):
        output_size = [int(s) for s in size]
        return F.affine_grid(theta, output_size, align_corners=self.align_corners)

    def extra_repr(self) -> str:
        return "align_corners={}".format(self.align_corners)
