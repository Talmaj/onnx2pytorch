import torch
import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator


def convert_onnx_pads_to_torch(pads):
    """
    Convert ONNX pads format to PyTorch F.pad format.

    ONNX: [begin_d0, begin_d1, ..., begin_dN-1, end_d0, end_d1, ..., end_dN-1]
    PyTorch: [left, right, top, bottom, front, back, ...] (last dim first, begin/end interleaved)
    """
    pad_dim = len(pads) // 2
    if pad_dim == 0:
        return []

    # Split into begins and ends, then interleave in reversed dimension order
    begins = pads[:pad_dim]
    ends = pads[pad_dim:]

    # Interleave begin and end for each dimension, then reverse dimension order
    result = []
    for i in range(pad_dim - 1, -1, -1):
        result.extend([begins[i], ends[i]])

    return result


class Pad(Operator):
    def __init__(self, mode="constant", padding=None):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            # Static padding already converted at construction time
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        else:
            # Dynamic pads - need to convert from ONNX to PyTorch format
            if isinstance(pads, torch.Tensor):
                pads = pads.tolist()
            else:
                pads = [
                    int(p.item()) if isinstance(p, torch.Tensor) else int(p)
                    for p in pads
                ]
            # Convert ONNX format to PyTorch format
            pads = convert_onnx_pads_to_torch(pads)

        # Convert value to float if it's a scalar tensor
        # ONNX spec requires constant_value to be a scalar
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                # Non-scalar value is invalid per ONNX spec, use default
                value = 0

        out = F.pad(input, pads, mode=self.mode, value=value)
        return out

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
