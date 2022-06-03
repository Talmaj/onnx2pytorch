import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator


class Pad(Operator):
    def __init__(self, mode="constant", padding=None, value=0):
        self.mode = mode
        self.padding = padding
        self.value = value
        super().__init__()

    def forward(self, input, pads=None):
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        out = F.pad(input, list(pads), mode=self.mode, value=self.value)
        return out

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
