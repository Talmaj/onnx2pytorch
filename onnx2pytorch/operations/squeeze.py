import torch

from onnx2pytorch.operations.base import Operator
from onnx2pytorch.utils import get_selection


class Squeeze(Operator):
    def __init__(self, dim=None):
        self.dim = dim
        if dim is not None:
            self.selection = get_selection(torch.tensor(0), self.dim)
        super().__init__()

    def forward(self, input):
        if self.dim is None:
            return torch.squeeze(input)
        else:
            return input[self.selection]
