import torch
from torch import nn

from onnx2pytorch.utils import as_input_dtype, get_reduce_dims


class ReduceSumSquare(nn.Module):
    """
    Computes the sum of the squared elements of the input tensor's elements along the provided axes.

    Equivalent to ReduceSum(Square(data), axes, keepdim).
    """

    def __init__(
        self, opset_version, dim=None, keepdim=True, noop_with_empty_axes=False
    ):
        self.opset_version = opset_version
        self.dim = dim
        self.keepdim = bool(keepdim)
        self.noop_with_empty_axes = noop_with_empty_axes
        super().__init__()

    def forward(self, data: torch.Tensor, axes: torch.Tensor = None):
        dim = get_reduce_dims(data, self.dim, axes, self.noop_with_empty_axes)
        if dim is None:
            return torch.square(data)
        ret = torch.sum(torch.square(data), dim=dim, keepdim=self.keepdim)
        return as_input_dtype(ret, data)
