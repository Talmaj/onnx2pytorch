import torch
from torch.nn.functional import one_hot

from onnx2pytorch.operations.base import Operator


class OneHot(Operator):
    def __init__(self, dim=-1, non_zero_values_only=False):
        self.dim = dim
        self.non_zero_values_only = non_zero_values_only
        super().__init__()

    def forward(self, indices, depth, values):
        if self.non_zero_values_only:
            off_value, on_value = -1, 1
        else:
            off_value, on_value = values
        depth = int(depth)

        # ONNX counts a negative index from the end and leaves an out of range
        # one entirely off, while one_hot rejects both.
        indices = indices.to(torch.int64)
        wrapped = torch.where(indices < 0, indices + depth, indices)
        in_range = (wrapped >= 0) & (wrapped < depth)
        out = one_hot(wrapped * in_range, depth) * in_range.unsqueeze(-1)
        out = out * (on_value - off_value) + off_value
        if not self.non_zero_values_only:
            out = out.to(values.dtype)

        rank = indices.dim()
        dim = self.dim + rank + 1 if self.dim < 0 else self.dim
        if not rank == dim:  # permute only if dim not last dimension
            order = list(range(rank))
            order.insert(dim, -1)
            out = out.permute(order)
        return out
