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
        if isinstance(dims, int):
            dims = [dims]
        # The axes count against the input rank, so resolve them before removing
        # anything and then work from the back, which keeps the lower axes valid.
        rank = input.dim()
        for dim in sorted((int(d) % rank for d in dims), reverse=True):
            input = torch.squeeze(input, dim=dim)
        return input
