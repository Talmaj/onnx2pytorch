import torch
import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator
from onnx2pytorch.utils import extract_padding_params

TORCH_MODES = {
    "constant": "constant",
    "edge": "replicate",
    "reflect": "reflect",
    "wrap": "circular",
}


def _to_int_list(values):
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.tolist()]
    return [int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in values]


def _expand_pads_to_axes(pads, axes, rank):
    """Spread pads given for a subset of axes over all axes."""
    full = [0] * (2 * rank)
    half = len(pads) // 2
    for i, axis in enumerate(axes):
        axis %= rank
        full[axis] = pads[i]
        full[rank + axis] = pads[half + i]
    return full


class Pad(Operator):
    def __init__(self, mode="constant", padding=None, value=0):
        if mode not in TORCH_MODES:
            raise NotImplementedError("Pad mode {} not implemented.".format(mode))
        self.mode = TORCH_MODES[mode]
        self.padding = padding
        self.value = value
        super().__init__()

    def forward(self, input, pads=None, value=None, axes=None):
        if self.padding is not None:
            # Already converted to torch convention at construction time.
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        else:
            pads = _to_int_list(pads)
            if axes is not None:
                pads = _expand_pads_to_axes(pads, _to_int_list(axes), input.dim())
            pads = extract_padding_params(pads)

        if value is None:
            value = self.value
        # F.pad only accepts a python scalar, while ONNX passes constant_value
        # as a 0-d tensor.
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else 0

        if self.mode == "constant":
            return F.pad(input, pads, mode=self.mode, value=value)

        # The other torch modes want the padded axes preceded by a batch and a
        # channel axis, ONNX pads tensors of any rank.
        missing = max(0, len(pads) // 2 + 2 - input.dim())
        padded = F.pad(input[(None,) * missing], pads, mode=self.mode)
        return padded[(0,) * missing]

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
