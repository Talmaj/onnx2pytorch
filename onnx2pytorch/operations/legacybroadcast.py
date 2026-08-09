import torch
from torch import nn


class LegacyBroadcast(nn.Module):
    """
    Pre-7 broadcasting of the arithmetic operators.

    B is aligned with A starting at dimension axis instead of at the last
    dimension, which is what torch and numpy do.
    """

    def __init__(self, op, axis=None):
        super().__init__()
        self.op = op
        self.axis = axis

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        if self.axis is not None and B.ndim:
            axis = self.axis + A.ndim if self.axis < 0 else self.axis
            shape = list(B.shape) + [1] * (A.ndim - axis - B.ndim)
            B = B.reshape(shape)
        return self.op(A, B)

    def extra_repr(self) -> str:
        return "op={}, axis={}".format(self.op.__name__, self.axis)
