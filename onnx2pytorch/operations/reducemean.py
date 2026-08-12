import torch
from torch import nn

from onnx2pytorch.utils import as_input_dtype, get_reduce_dims


class ReduceMean(nn.Module):
    def __init__(self, dim=None, keepdim=True, noop_with_empty_axes=False):
        self.dim = dim
        self.keepdim = bool(keepdim)
        self.noop_with_empty_axes = noop_with_empty_axes
        super().__init__()

    def forward(self, data: torch.Tensor, axes: torch.Tensor = None):
        dim = get_reduce_dims(data, self.dim, axes, self.noop_with_empty_axes)
        if dim is None:
            return data
        # torch.mean rejects integers, onnx truncates the mean back to them
        values = data if data.dtype.is_floating_point else data.double()
        return as_input_dtype(torch.mean(values, dim=dim, keepdim=self.keepdim), data)
