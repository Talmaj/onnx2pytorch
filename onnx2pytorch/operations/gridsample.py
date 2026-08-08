import torch
from torch import nn
from torch.nn import functional as F

MODES = {
    "bilinear": "bilinear",
    "linear": "bilinear",
    "nearest": "nearest",
    "bicubic": "bicubic",
    "cubic": "bicubic",
}


class GridSample(nn.Module):
    def __init__(self, mode="linear", padding_mode="zeros", align_corners=0):
        super().__init__()
        if mode not in MODES:
            raise NotImplementedError(
                "GridSample mode={} not implemented.".format(mode)
            )
        self.mode = MODES[mode]
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)

    def forward(self, X: torch.Tensor, grid: torch.Tensor):
        return F.grid_sample(
            X,
            grid.to(X.dtype),
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

    def extra_repr(self) -> str:
        return "mode={}, padding_mode={}, align_corners={}".format(
            self.mode, self.padding_mode, self.align_corners
        )
