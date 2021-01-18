import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator


class Pad(Operator):
    def __init__(self, mode="constant"):
        self.mode = mode
        super().__init__()

    def forward(self, input, pads, value=0):
        return F.pad(input, pads.tolist(), mode=self.mode, value=value)
