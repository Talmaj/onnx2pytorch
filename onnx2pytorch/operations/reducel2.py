import torch
from torch import nn

from onnx2pytorch.utils import as_input_dtype, get_reduce_dims


class ReduceL2(nn.Module):
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
            return torch.abs(data)
        squares = torch.square(data)
        if not data.dtype.is_floating_point:
            # An integer square sum has to be rooted in floating point
            squares = squares.double()
        ret = torch.sqrt(torch.sum(squares, dim=dim, keepdim=self.keepdim))
        return as_input_dtype(ret, data)
