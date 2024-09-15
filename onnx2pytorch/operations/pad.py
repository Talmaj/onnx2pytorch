import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator


def preprocess_pads(in_activations):
    """
    If pads is 8-d array for 4d input or pads is 6-d array for 3d input.
    Convert pads from [b1,b2,...,e1,e2,...] to [b1,e1,b2,e2,...]

    """
    input = in_activations[0]
    pads = list(in_activations[1])
    if len(pads)//2 == len(input.size()):
        import torch
        new_pads = []
        mid_idx = len(pads)//2
        pads.reverse()
        for i in range(mid_idx, len(pads)):
            new_pads.append(pads[i])
            new_pads.append(pads[i-mid_idx])
        in_activations[1] = torch.tensor(new_pads)
    return in_activations


class Pad(Operator):
    def __init__(self, mode="constant", padding=None):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        out = F.pad(input, list(pads), mode=self.mode, value=value)
        return out

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
