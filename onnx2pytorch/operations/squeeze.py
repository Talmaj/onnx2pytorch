import torch

from onnx2pytorch.operations.base import Operator
from onnx2pytorch.utils import get_selection


class Squeeze(Operator):
    def __init__(self, opset_version, dim=None):
        self.opset_version = opset_version
        self.dim = dim
        super().__init__()

    def forward(self, input: torch.Tensor, axes: torch.Tensor = None):
        if self.opset_version < 13:
            dims = self.dim
        else:
            dims = axes

        if dims is None:
            return torch.squeeze(input)
        elif isinstance(dims, int):
            return torch.squeeze(input, dim=dims)
        else:
            for dim in sorted(dims, reverse=True):
                input = torch.squeeze(input, dim=dim)
            return input
