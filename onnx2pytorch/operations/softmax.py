import torch
from torch import nn
from torch.nn import functional as F


def coerce_to_2d(input, axis):
    """Collapse the input around axis the way the pre-13 operators specify."""
    if axis < 0:
        axis += input.ndim
    rows = 1
    for size in input.shape[:axis]:
        rows *= size
    return input.reshape(rows, -1)


class NormalizingOperator(nn.Module):
    """
    Base for Softmax, LogSoftmax and Hardmax.

    Before opset 13 these coerce their input to 2D around axis, normalize the
    rows and restore the original shape. From opset 13 on they normalize the
    axis in place, and the default axis changed from 1 to -1.
    """

    def __init__(self, opset_version, dim=None):
        super().__init__()
        self.opset_version = opset_version
        self.dim = dim if dim is not None else (-1 if opset_version >= 13 else 1)

    def normalize(self, input, dim):
        raise NotImplementedError

    def forward(self, input: torch.Tensor):
        if self.opset_version >= 13:
            return self.normalize(input, self.dim)
        return self.normalize(coerce_to_2d(input, self.dim), 1).reshape(input.shape)

    def extra_repr(self) -> str:
        return "dim={}".format(self.dim)


class Softmax(NormalizingOperator):
    def normalize(self, input, dim):
        return F.softmax(input, dim=dim)


class LogSoftmax(NormalizingOperator):
    def normalize(self, input, dim):
        return F.log_softmax(input, dim=dim)
