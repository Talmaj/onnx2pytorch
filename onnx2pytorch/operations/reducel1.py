import torch
from torch import nn

from onnx2pytorch.utils import as_input_dtype, get_reduce_dims


class ReduceL1(nn.Module):
    def __init__(self, dim=None, keepdim=True, noop_with_empty_axes=False):
        self.dim = dim
        self.keepdim = bool(keepdim)
        self.noop_with_empty_axes = noop_with_empty_axes
        super().__init__()

    def forward(self, data: torch.Tensor, axes: torch.Tensor = None):
        dim = get_reduce_dims(data, self.dim, axes, self.noop_with_empty_axes)
        if dim is None:
            return torch.abs(data)
        ret = torch.sum(torch.abs(data), dim=dim, keepdim=self.keepdim)
        return as_input_dtype(ret, data)
