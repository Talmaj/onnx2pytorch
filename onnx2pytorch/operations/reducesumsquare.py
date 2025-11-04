import torch
from torch import nn


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
        # In opset < 13, axes is an attribute (self.dim)
        # In opset >= 13, axes is an optional input
        if self.opset_version < 13:
            dims = self.dim
        else:
            dims = axes

        if dims is None:
            if self.noop_with_empty_axes:
                return data
            else:
                # Reduce over all dimensions
                dims = tuple(range(data.ndim))

        if isinstance(dims, int):
            dim = dims
        else:
            dim = tuple(list(dims))

        # Compute sum of squares: sum(x^2)
        ret = torch.sum(torch.square(data), dim=dim, keepdim=self.keepdim)
        return ret
