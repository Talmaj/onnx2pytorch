import torch
from torch import nn


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
        if self.opset_version < 13:
            dims = self.dim
        else:
            dims = axes
        if dims is None:
            if self.noop_with_empty_axes:
                return data
            else:
                dims = tuple(range(data.ndim))
        if isinstance(dims, int):
            return torch.sum(data, dim=dims, keepdim=self.keepdim)
        else:
            return torch.sum(data, dim=tuple(list(dims)), keepdim=self.keepdim)
