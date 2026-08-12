import torch
from torch import nn

from onnx2pytorch.operations.base import Operator


class Unsqueeze(Operator):
    def __init__(self, opset_version, dim=None):
        self.opset_version = opset_version
        self.dim = dim
        super().__init__()

    def forward(self, data: torch.Tensor, axes: torch.Tensor = None):
        if self.opset_version < 13:
            dims = self.dim
        else:
            dims = axes
        if dims is None:
            raise ValueError("Unsqueeze expects axes")
        if isinstance(dims, int):
            dims = [dims]
        # Each axis counts against the output rank, so resolve them all against
        # it up front. Inserting in ascending order then leaves each remaining
        # axis pointing at the position it names in the finished tensor.
        rank = data.dim() + len(dims)
        for dim in sorted(int(d) % rank for d in dims):
            data = torch.unsqueeze(data, dim=dim)
        return data
