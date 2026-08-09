import torch
import torch.nn.functional as F
from torch import nn


class LRN(nn.Module):
    """
    Local Response Normalization across channels.

    Not delegated to nn.LocalResponseNorm because that one centres an even
    sized window on the other side of the channel than onnx does.
    """

    def __init__(self, alpha=0.0001, beta=0.75, bias=1.0, size=None):
        super().__init__()
        if size is None:
            raise ValueError("size parameter is required for LRN operation")

        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        self.size = size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        before = (self.size - 1) // 2
        squares = X.square().movedim(1, -1)
        squares = F.pad(squares, (before, self.size - 1 - before))
        summed = squares.unfold(-1, self.size, 1).sum(-1).movedim(-1, 1)
        return X / (self.bias + self.alpha / self.size * summed).pow(self.beta)

    def extra_repr(self) -> str:
        return "alpha={}, beta={}, bias={}, size={}".format(
            self.alpha, self.beta, self.bias, self.size
        )
