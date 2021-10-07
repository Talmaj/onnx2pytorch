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
            dims = torch.Size(axes)
        if dims is None:
            raise ValueError("Unsqueeze expects axes")
        elif isinstance(dims, int):
            return torch.unsqueeze(data, dim=dims)
        else:
            for dim in sorted(dims):
                data = torch.unsqueeze(data, dim=dim)
            return data
