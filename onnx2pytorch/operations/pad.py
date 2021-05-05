import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator
from onnx2pytorch.utils import extract_padding_params


class Pad(Operator):
    def __init__(self, mode="constant", padding=None):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            pads = self.padding
        elif pads is not None:
            pads = extract_padding_params(pads)
        else:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        return F.pad(input, list(pads), mode=self.mode, value=value)

    def extra_repr(self) -> str:
        return "padding={}".format(self.padding)
