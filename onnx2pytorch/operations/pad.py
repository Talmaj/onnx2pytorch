import torch
import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator
from onnx2pytorch.utils import convert_onnx_pads_to_torch


class Pad(Operator):
    def __init__(self, mode="constant", padding=None):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            # Already converted to torch convention at construction time.
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        else:
            if isinstance(pads, torch.Tensor):
                pads = pads.tolist()
            else:
                pads = [
                    int(p.item()) if isinstance(p, torch.Tensor) else int(p)
                    for p in pads
                ]
            pads = convert_onnx_pads_to_torch(pads)

        # F.pad only accepts a python scalar, while ONNX passes constant_value
        # as a 0-d tensor.
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else 0

        out = F.pad(input, pads, mode=self.mode, value=value)
        return out

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
