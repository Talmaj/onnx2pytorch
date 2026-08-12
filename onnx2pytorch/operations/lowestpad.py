import torch.nn.functional as F
from torch import nn

from onnx2pytorch.utils import lowest_value


class LowestPad(nn.Module):
    """
    Pad with the lowest value the input dtype holds, so that max pooling ignores
    the pads. A literal -inf cannot be used, the input may well be an integer.
    """

    def __init__(self, padding):
        super().__init__()
        self.padding = tuple(padding)

    def forward(self, input):
        return F.pad(input, self.padding, value=lowest_value(input.dtype))

    def extra_repr(self) -> str:
        return "padding={}".format(self.padding)
