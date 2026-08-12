import torch
from torch import nn

from onnx2pytorch.utils import as_input_dtype, get_reduce_dims


class ReduceSum(nn.Module):
    def __init__(
        self, opset_version, dim=None, keepdim=True, noop_with_empty_axes=False
    ):
        self.opset_version = opset_version
        self.dim = dim
        self.keepdim = keepdim
        self.noop_with_empty_axes = noop_with_empty_axes
        super().__init__()

    def forward(self, data: torch.Tensor, axes: torch.Tensor = None):
        dim = get_reduce_dims(data, self.dim, axes, self.noop_with_empty_axes)
        if dim is None:
            return data
        return as_input_dtype(torch.sum(data, dim=dim, keepdim=self.keepdim), data)
